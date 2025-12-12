#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[1]:


# suppress warning outputs for using lal in jupuyter notebook
import warnings 
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import GWFish.modules as gw
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
import json
import os
import pandas as pd
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u


# In[2]:


def _wrap_ra_dec_pair(ra, dec):
    """Return (ra, dec) with RA in [0, 2π) and DEC in [-π/2, π/2],
    applying RA→RA+π when DEC is folded to preserve the sky position."""
    two_pi = 2*np.pi
    pi = np.pi

    ra = np.mod(ra, two_pi) #This puts RA within the domain [0,2π]

    # fold dec into [-π/2, π/2] with corresponding RA shift by π
    over  = dec >  pi/2
    under = dec < -pi/2

    # reflect
    ra   = np.where(over,  ra + pi, ra) #In this process, RA can add up to more than π
    dec  = np.where(over,  pi - dec, dec)  

    ra   = np.where(under, ra + pi, ra)
    dec  = np.where(under, -pi - dec, dec) #In this process, RA can add up to more than π

    ra = np.mod(ra, two_pi) #Here RA is renormalised to [0,2π]
    return ra, dec


# In[3]:


def wrap_angles_inplace(samples, idx):
    """
    Wrap angles in-place for the columns present in `idx`.
    Conventions used:
      - ra  ∈ [0, 2π)
      - dec ∈ [-π/2, π/2] (and RA is shifted by π when DEC is folded)
      - theta_jn ∈ [0, π]
      - psi ∈ [0, π)
    `samples` can be shape (D,) or (N, D).
    """
    two_pi = 2*np.pi
    pi = np.pi

    samples = np.asarray(samples)

    # Handle both 1D (single sample) and 2D (batch)
    if samples.ndim == 1:
        if ('ra' in idx) and ('dec' in idx):
            ra  = samples[idx['ra']] #Takes the value from the sample
            dec = samples[idx['dec']]
            ra, dec = _wrap_ra_dec_pair(ra, dec) #Wraps the values within the domain
            samples[idx['ra']]  = ra #Returns the wraped value to the sample set
            samples[idx['dec']] = dec

        if 'psi' in idx:
            samples[idx['psi']] = np.mod(samples[idx['psi']], pi) #Returns the wrapped value to the sample set

        if 'theta_jn' in idx:
            x = np.mod(samples[idx['theta_jn']], two_pi)
            samples[idx['theta_jn']] = x if x <= pi else (two_pi - x) #Returns the wrapped value to the sample set

        if 'phase' in idx:
            samples[idx['phase']] = np.mod(samples[idx['phase']], 2*np.pi) #Returns the wrapped value to the sample set

    elif samples.ndim == 2:
        if ('ra' in idx) and ('dec' in idx):
            ra_new, dec_new = _wrap_ra_dec_pair(samples[:, idx['ra']], samples[:, idx['dec']])
            samples[:, idx['ra']]  = ra_new
            samples[:, idx['dec']] = dec_new

        if 'psi' in idx:
            samples[:, idx['psi']] = np.mod(samples[:, idx['psi']], pi)

        if 'theta_jn' in idx:
            x = np.mod(samples[:, idx['theta_jn']], two_pi)
            samples[:, idx['theta_jn']] = np.where(x <= pi, x, two_pi - x)

        if 'phase' in idx:
            samples[:, idx['phase']] = np.mod(samples[:, idx['phase']], 2*np.pi)

    return samples


# In[4]:


def physical_mask(samples, idx, zmax=1000):
    """
    Returns a boolean mask selecting rows that satisfy:
      mass_1 > mass_2 > 0
      luminosity_distance > 0
      0 <= a_1 <= 1, 0 <= a_2 <= 1
    Missing keys are ignored (no filtering on that criterion).
    """
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s.reshape(1, -1)

    m = np.ones(s.shape[0], dtype=bool)

    if 'mass_1_source' in idx and 'mass_2_source' in idx:
        m1 = s[:, idx['mass_1_source']]
        m2 = s[:, idx['mass_2_source']]
        m &= (m1 > m2) & (m2 > 0)

    if 'luminosity_distance' in idx:
        dl = s[:, idx['luminosity_distance']]
        dl_max = float(Planck18.luminosity_distance(zmax).to('Mpc').value)
        m &= (dl > 0) & (dl < dl_max)

    if 'a_1' in idx:
        a1 = s[:, idx['a_1']]
        m &= (a1 >= 0) & (a1 <= 1)

    if 'a_2' in idx:
        a2 = s[:, idx['a_2']]
        m &= (a2 >= 0) & (a2 <= 1)

    return m


# In[5]:


#Parameters for the catalog
PARAMS15 = ['mass_1_source', 'mass_2_source', 'q', 'redshift', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
                     'psi', 'phase', 'geocent_time', 'a_1', 'a_2','tilt_1','tilt_2'] # 15 parameters

PARAMS13 = ['mass_1_source', 'mass_2_source', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
                     'psi', 'phase', 'geocent_time', 'a_1', 'a_2','tilt_1','tilt_2'] # 13 parameters

PARAMS11 = ['mass_1_source', 'mass_2_source', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
                     'psi', 'phase', 'geocent_time', 'a_1', 'a_2'] # 11 parameters


# In[6]:


#Drop the columns in the error file that correspond to unwanted columns/parameters for the sampling from the multivariate Gaussian

#drop_errorfile = {0,3,4,6,9,10,14,15} #skips SNR, q, redshift, theta_jn, psi, phase, tilt1, tilt2
drop_errorfile = {0,3,4,14,15} #skips SNR, q, redshift, tilt1, tilt2
keep = [i for i in range(16) if i not in drop_errorfile] #0->15
means11 = np.loadtxt('Errors_LISA_MBHB_catalog_popIII_withfisher_tilts_Msc_B23_zW21_PhenomPv2_SNR0.txt', skiprows=1, usecols=keep)


# In[7]:


# Load covariances (inverse Fisher) and base (true) params
invfisher_13d = np.load("inv_fisher_matrices_LISA_MBHB_catalog_popIII_withfisher_tilts_Msc_B23_zW21_PhenomPv2_SNR0.npy")  # shape (N, 13, 13)

#Drop unwanted rows/column indexes in the 13-d inverse fisher matrix for the sampling from the multivariate Gaussian
#drop_invfisher = {3,6,7,11,12} #skips theta_jn, psi, phase, tilt1, tilt2
drop_invfisher = {11,12} #skips tilt1, tilt2
keep = [i for i in range(13) if i not in drop_invfisher] #0->12 

N = invfisher_13d.shape[0]
invfisher_red = np.empty((N, len(keep), len(keep)))
for e in range(N):
    invfisher_red[e] = invfisher_13d[e][np.ix_(keep, keep)]


# In[8]:


def draw_one_with_constraints(mu, Sigma, rng, PARAMS11,
                              N=10):      
    S = Sigma
    #theta = rng.multivariate_normal(mu, S, size=N, check_valid="raise")
    
    try:
        theta = rng.multivariate_normal(mu, S, size=N, check_valid="raise")
    except Exception:
        w, V = np.linalg.eigh(S); w = np.clip(w, 1e-18, None)
        Z = rng.standard_normal((N, mu.size))
        theta = mu + Z @ (V @ np.diag(np.sqrt(w))).T

    idx = {p:i for i,p in enumerate(PARAMS11)}
    
    wrap_angles_inplace(theta, idx)              # 1) wrap angles in-place
    mask = physical_mask(theta, idx)             # 2) build mask for physical constraints
    valid_samples = theta[mask]                  #    keep only physical rows

    print(f"Kept {mask.sum()} / {len(mask)} samples.")

    # If you want just one valid sample (e.g., the first one):
    first_valid = valid_samples[0] if len(valid_samples) else None
    
    return first_valid


# In[9]:


N_candidates = 5000
samples11 = np.empty_like(means11)  # means11: (n_events, 11)
rng = np.random.default_rng(42626)

for i in range(len(means11)):
    samples11[i] = draw_one_with_constraints(means11[i], invfisher_red[i], rng, PARAMS11,
                                             N=N_candidates)  


# In[10]:

# Make a labeled DataFrame
parameters = pd.DataFrame(samples11, columns=PARAMS11)
parameters


# In[11]:


#Parameters when skipping only tilt1, and tilt2 from the multivariate sampling (including theta_jn, psi, phase in the multivariate)
tilt_1   = np.random.uniform(0, np.pi, N)
tilt_2   = np.random.uniform(0, np.pi, N)


# In[12]:


inj15 = np.empty((N, len(PARAMS15)), dtype=float)
dL = samples11[:, 2] * u.Mpc
z_vals = np.array([z_at_value(Planck18.luminosity_distance, d, zmin=1e-9, zmax=1000)
                   for d in dL], dtype=float)

"""
#Parameters when skipping tilt1, and tilt2 from the multivariate sampling
inj13[:, 0]  = samples11[:, 0]                  # mass_1
inj13[:, 1]  = samples11[:, 1]                  # mass_2
inj13[:, 2]  = samples11[:, 2]                  # d_L
inj13[:, 3]  = samples11[:, 3]                  # theta_jn
inj13[:, 4]  = samples11[:, 4]                  # ra
inj13[:, 5]  = samples11[:, 5]                  # dec
inj13[:, 6]  = samples11[:, 6]                  # psi 
inj13[:, 7]  = samples11[:, 7]                  # phase
inj13[:, 8]  = samples11[:, 8]                  # geocent_time
inj13[:, 9]  = samples11[:, 9]                  # a_1
inj13[:,10]  = samples11[:, 10]                 # a_2
inj13[:,11]  = tilt_1                            # tilt_1  
inj13[:,12]  = tilt_2                            # tilt_2 
"""

#Parameters when skipping tilt1, and tilt2 from the multivariate sampling
inj15[:, 0]  = samples11[:, 0]                  # mass_1
inj15[:, 1]  = samples11[:, 1]                  # mass_2
inj15[:, 2]  = samples11[:, 1]/samples11[:, 0]  # q
inj15[:, 3]  = z_vals                           # z
inj15[:, 4]  = samples11[:, 2]                  # d_L
inj15[:, 5]  = samples11[:, 3]                  # theta_jn
inj15[:, 6]  = samples11[:, 4]                  # ra
inj15[:, 7]  = samples11[:, 5]                  # dec
inj15[:, 8]  = samples11[:, 6]                  # psi 
inj15[:, 9]  = samples11[:, 7]                  # phase
inj15[:, 10]  = samples11[:, 8]                  # geocent_time
inj15[:, 11]  = samples11[:, 9]                  # a_1
inj15[:,12]  = samples11[:, 10]                 # a_2
inj15[:,13]  = tilt_1                            # tilt_1  
inj15[:,14]  = tilt_2                            # tilt_2 


# In[13]:


import pandas as pd

# Make a labeled DataFrame (recommended for GWFish interface)
parameters = pd.DataFrame(inj15, columns=PARAMS15)
parameters


# In[14]:


#f = pd.DataFrame(parameters)
#df.to_csv("mbhb_popIIInoisy_catalog_Mz.tsv", sep="\t", index=False)


# In[15]:


# We choose a waveform approximant suitable for BNS analysis
# In this case we are taking into account tidal polarizability effects
waveform_model = 'IMRPhenomPv2'
f_ref = 1e-4


# In[16]:


# Choose the detector onto which you want to project the signal
detector = 'LISA'

# The following function outputs the signal projected onto the chosen detector
signal, _ = gw.utilities.get_fd_signal(parameters, detector, waveform_model, f_ref) # waveform_model and f_ref are passed together
frequency = gw.detection.Detector(detector).frequencyvector[:, 0]


# In[17]:


# add the detector's sensitivity curve and plot the characteristic strain
psd_data = gw.utilities.get_detector_psd(detector)


# In[18]:


# Plot the time before the merger as a function of the frequency
_, t_of_f = gw.utilities.get_fd_signal(parameters, detector, waveform_model, f_ref)


# In[19]:


convert_from_seconds_to_hours = 3600


# ## Calculate SNR

# In[20]:


# The networks are the combinations of detectors that will be used for the analysis
# The detection_SNR is the minimum SNR for a detection:
#   --> The first entry specifies the minimum SNR for a detection in a single detector
#   --> The second entry specifies the minimum network SNR for a detection
detectors = ['LISA']
network = gw.detection.Network(detector_ids = detectors, detection_SNR = (0, 12.))
snr = gw.utilities.get_snr(parameters, network, waveform_model, f_ref)


# In[21]:


#df = pd.DataFrame(snr)
#df.to_csv("mbhb_popIIInoisy_catalog_snr_Mz.tsv", sep="\t", index=False)


# ## Calculate $1\sigma$ Errors
# For a more realistic analysis we can include the **duty cycle** of the detectors using `use_duty_cycle = True`

# In[22]:


# The fisher parameters are the parameters that will be used to calculate the Fisher matrix
# and on which we will calculate the errors

fisher_parameters = ['mass_1_source', 'mass_2_source', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
                     'psi', 'phase', 'geocent_time', 'a_1', 'a_2', 'tilt_1', 'tilt_2']

#fisher_parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 
#                     'psi', 'phase', 'a_1', 'a_2']


# In[ ]:


detected, network_snr, parameter_errors, sky_localization = gw.fishermatrix.compute_network_errors(
        network = gw.detection.Network(detector_ids = ['LISA'], detection_SNR = (0., 12.)),
        parameter_values = parameters,
        fisher_parameters=fisher_parameters, 
        waveform_model = waveform_model,
        f_ref = 1e-4,
        eps=1e-5,
        eps_mass=1e-5,
        )   
        # use_duty_cycle = False, # default is False anyway
save_matrices = True, # default is False anyway, put True if you want Fisher and covariance matrices in the output
save_matrices_path = '/home/2809904g/popIII', # default is None anyway,
                                     # otherwise specify the folder
                                     # where to save the Fisher and
                                     # corresponding covariance matrices
    
    


# In[ ]:


# Choose percentile factor of sky localization and pass from rad2 to deg2
percentile = 90.
sky_localization_90cl = sky_localization * gw.fishermatrix.sky_localization_percentile_factor(percentile)
#sky_localization_90cl


# In[ ]:


# One can create a dictionary with the parameter errors, the order is the same as the one given in fisher_parameters
parameter_errors_dict = {}
for i, parameter in enumerate(fisher_parameters):
    parameter_errors_dict['err_' + parameter] = np.squeeze(parameter_errors)[i]

#print('The parameter errors of the event are ')
parameter_errors_dict


# In[ ]:


data_folder = '/home/2809904g/popIII' 
network = gw.detection.Network(detector_ids = ['LISA'], detection_SNR = (0., 12.))
gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                        parameter_values  = parameters,
                                        fisher_parameters = fisher_parameters, 
                                        sub_network_ids_list = [[0]],
                                        population_name = f'MBHB_catalog_popIII_noisy_signal_with_fishertilts_Msc_B23_zW21_PhenomPv2',
                                        waveform_model = waveform_model,
                                        f_ref = 1e-4,
                                        save_path = data_folder,
                                        save_matrices = True,
                                        eps=1e-5,
                                        eps_mass=1e-5
                                        #decimal_output_format='%.6E'
                                        )


# In[ ]:




