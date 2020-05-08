import torch
import numpy as np

def compute_complexity_measures(data):
    # data is a dictionary containing the fields
    # 'sum_sqr_frobenius_norm', 'sum_sqr_nuclear_norm', 'sum_sqr_spectral_norm', and 't'

    t = np.array(data['t'])
    delta_t = t[1:]-t[0:-1]

    ss_fro = np.array(data['sum_sqr_frobenius_norm'])
    ss_fro = 0.5*(ss_fro[0:-1]+ss_fro[1:]) # midpoint

    log2_frobenius = np.sum(delta_t*np.log2(ss_fro))

    ss_nuc = np.array(data['sum_sqr_nuclear_norm'])
    ss_nuc = 0.5 * (ss_nuc[0:-1] + ss_nuc[1:])  # midpoint

    log2_nuclear = np.sum(delta_t * np.log2(ss_nuc))

    ss_spe = np.array(data['sum_sqr_spectral_norm'])
    ss_spe = 0.5 * (ss_spe[0:-1] + ss_spe[1:])  # midpoint

    log2_spectral = np.sum(delta_t * np.log2(ss_spe))

    log_complexity_measures = dict()
    log_complexity_measures['log2_frobenius'] = log2_frobenius
    log_complexity_measures['log2_nuclear'] = log2_nuclear
    log_complexity_measures['log2_spectral'] = log2_spectral

    return log_complexity_measures