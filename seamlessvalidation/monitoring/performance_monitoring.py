from ..validation.keymetrics import (
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

from sklearn.base import ClassifierMixin, RegressorMixin,BaseEstimator, ClusterMixin
from sklearn.model_selection import (cross_val_score, cross_val_predict, StratifiedKFold,LeaveOneOut, KFold)
from seamlessvalidation import pd

from .data_drift import (monitor_data_drift)


def monitor_performance(model, X, y,**kwargs):
    data_compare=kwargs.get('data_compare', None)
    data_new=kwargs.get('data_new', None)
    p_value=kwargs.get('p_value', 0.05)

    mean_accuracy_or_r2 = None
    mae = mse = rmse = roc_auc =precision=recall=f1=precision_recall_auc = None
    cm = None

    #roc_auc = calculate_roc_auc(y_true, y_pred)
    #cm = calculate_confusion_matrix(y_true, y_pred)
    #precision = calculate_precision(y_true, y_pred)
    #recall = calculate_recall(y_true, y_pred)
    #f1 = calculate_f1_score(y_true, y_pred)

    #y_pred = cross_val_predict(model, X, y)
    #y_true=y
    #roc_auc, precision_recall_auc=calculate_roc_auc(model, X, y)

    # Check if the task is a regression problem
    is_regressor = all(isinstance(val, (int, float)) for val in y)
    # Check if the model is a classifier or a regressor
    #is_classifier = isinstance(model, ClassifierMixin)
    #is_regressor = isinstance(model, RegressorMixin)
    if not is_regressor:
        roc_auc = calculate_roc_auc(model, X, y)
        cm=calculate_confusion_matrix(model, X, y)
        precision = calculate_precision(model, X, y)
        recall =calculate_recall(model, X, y)
        f1 = calculate_f1_score(model, X, y)
        
        mean_accuracy_or_r2 = calculate_mean_accuracy_or_r2(model, X, y)
        mae =  'Regression only'
        mse =  'Regression only'
        rmse =  'Regression only'


    # Include regression metrics if it's a regression problem
    else:
        mean_accuracy_or_r2 = calculate_mean_accuracy_or_r2(model, X, y)
        mae = calculate_mean_absolute_error(model, X, y)
        mse = calculate_mean_squared_error(model, X, y)
        rmse = calculate_root_mean_squared_error(model, X, y)

        roc_auc='Classification only'
        cm='Classification only'
        precision='Classification only'
        recall='Classification only'
        f1='Classification only'

    if data_compare is None or data_new is None:
        data_drift_result = 'Comparison data not provided'
    
    else:
        data_drift_result = monitor_data_drift(data_compare=data_compare, data_new=data_new,p_value=p_value)

    metrics = {
        "Data Drift Statistics" : data_drift_result,
        "ROC AUC": roc_auc,
        "Confusion Matrix": cm,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Mean Accuracy (Classifiers) or R^2 (Regressors)": mean_accuracy_or_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }
    
    
    return metrics