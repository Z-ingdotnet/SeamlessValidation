#from seamlessvalidation import pd
#from seamlessvalidation import np
#from seamlessvalidation import plt
#from seamlessvalidation import io

from sklearn.base import ClassifierMixin, RegressorMixin,BaseEstimator, ClusterMixin


from .keymetrics import (
    calculate_roc_auc,
    calculate_mean_accuracy_or_r2,
    calculate_confusion_matrix,
    calculate_mean_absolute_error,
    calculate_mean_squared_error,
    calculate_root_mean_squared_error,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_silhouette_score
)


from .allin1_validation import (
    cross_validation
)

from ..monitoring.performance_monitoring import *

from ..monitoring.data_drift import * #import (monitor_data_drift)

#from sklearn import metrics

#from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn.datasets import make_classification, make_regression
#from sklearn.base import ClassifierMixin, RegressorMixin,BaseEstimator, ClusterMixin
#from sklearn.model_selection import (cross_val_score, cross_val_predict, StratifiedKFold,LeaveOneOut, KFold)

#from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_curve,
#                             auc, mean_absolute_error, mean_squared_error, roc_auc_score,
#                             f1_score,recall_score,precision_score,silhouette_score)


# Function to check if the model is a clustering algorithm
def is_clustering_model(model):
    if not issubclass(model.__class__, BaseEstimator):
        return False
    if issubclass(model.__class__, ClusterMixin) or hasattr(model, "fit_predict"):
        return True
    return False


def seamlessvalidation(model,phase='validation',**kwargs):
    X=kwargs.get('X', None)
    y=kwargs.get('y', None)
    n_splits=kwargs.get('n_splits', None)
    cv_strategy=kwargs.get('cv_strategy', None)
    data_compare=kwargs.get('data_compare', None)
    data_new=kwargs.get('data_new', None)
    p_value=kwargs.get('p_value', 0.05)

    is_classifier = isinstance(model, ClassifierMixin)
    is_regressor = isinstance(model, RegressorMixin)
    is_clustering= is_clustering_model(model)
    if is_clustering:
        metric=calculate_silhouette_score(X, model)
        metrics_dict = {
        'silhouette score':metric}
    else:
        if phase=='validation':
            metrics_dict=cross_validation(model, X, y, n_splits, cv_strategy)
        else: 
            metrics_dict=monitor_performance(model, X, y, data_compare=data_compare, data_new=data_new)
    return metrics_dict
        