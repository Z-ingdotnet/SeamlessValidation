#import numpy as np
#import pandas as pd
#from scipy import stats

from ..validation.keymetrics import (
    ks_drift_detection,
    kl_divergence,
    js_divergence
)


def monitor_data_drift(data_compare, data_new, p_value=0.05):
    """
    Monitor data drift using Kolmogorov-Smirnov Test
    """
    ks_drift_result = ks_drift_detection(data_compare, data_new, p_value)
    kl_divergence_result = kl_divergence(data_compare, data_new)
    js_divergence_result = js_divergence(data_compare, data_new)

    return ks_drift_result,kl_divergence_result,js_divergence_result
            #ks_drift_result 
           #kl_divergence_result
            #,
            #kl_divergence_result
            #, 
            #js_divergence_result

#def ks_drift_detection(data_new, data_compare, p_value=0.05):
#    """
#    Kolmogorov-Smirnov Test 
#    """
#    # Flatten input arrays
#    data_new_flat = np.ravel(data_new)
#    data_compare_flat = np.ravel(data_compare)
#    # Perform the Kolmogorov-Smirnov test
#    ks_stat, p_val = stats.ks_2samp(data_new_flat, data_compare_flat)
#    # If the K-S statistic is small or the p-value is high (greater than the significance level, say 5%), then we cannot reject the hypothesis that the distributions of the two samples are the same.
#    if p_val < p_value: 
#        drift= 'Data Drift Detected, ks_drift:'
#    else:
#        drift= 'No Data Drift, ks_drift'
#    #return ks_stat 
#    return drift,ks_stat
##    if p_val < p_value: 
##        return 'Data Drift Detected'
##    else:
##        return 'No Data Drift'



#def kl_divergence(data_compare, data_new):
#    """
#    Calculate Kullback-Leibler divergence
#    """
#    kl_div = stats.entropy(data_compare, data_new)
#    return "kl_divg: " + str(kl_div)

#def js_divergence(data_compare, data_new):
#    """
#    Calculate Jensen-Shannon divergence
#    """
#    p = 0.5 * (data_compare + data_new)
#    js_div = 0.5 * (stats.entropy(data_compare, p) + stats.entropy(data_new, p))
#    return "js_divg: " + str(js_div)
