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

