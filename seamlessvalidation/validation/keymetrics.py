from seamlessvalidation import pd
from seamlessvalidation import np
from seamlessvalidation import plt
from seamlessvalidation import io


from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_curve,
                             auc, mean_absolute_error, mean_squared_error, roc_auc_score,
                             f1_score,recall_score,precision_score,silhouette_score)
from sklearn.model_selection import (cross_val_score, cross_val_predict, StratifiedKFold,LeaveOneOut, KFold)

from scipy import stats
#from scipy.stats import entropy

def calculate_roc_auc(model, X, y, n_splits=5):
    if hasattr(model, "predict_proba"):
        proba_predictions = cross_val_predict(model, X, y, cv=n_splits, method="predict_proba")
        if len(np.unique(y)) == 2:  # Only for binary classification
                roc_auc = roc_auc_score(y, proba_predictions[:, 1])
                precision, recall, _ = precision_recall_curve(y, proba_predictions[:, 1])
                pr_curve = (precision, recall)
                precision_recall_auc = auc(recall, precision)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid(True)
                #plt.show()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)  # Move to the start of the BytesIO object            
                #plt.close() 
                return roc_auc,precision_recall_auc
            #return roc_auc_score(y, proba_predictions[:, 1])
    return None

def calculate_mean_accuracy_or_r2(model, X, y, n_splits=5, scoring='accuracy'):
    #scoring_method = 'accuracy' if is_classifier else 'r2'
    scoring_method = 'r2' if all(isinstance(val, (int, float)) for val in y) else 'accuracy'
    return cross_val_score(model, X, y, cv=n_splits, scoring=scoring_method).mean()

#from sklearn.metrics import r2_score
#from sklearn.metrics import accuracy_score
#def calculate_mean_accuracy_or_r2(model, X, y):
#    if all(isinstance(val, (int, float)) for val in y):  # Regression task
#        return r2_score(y, model.predict(X))
#    else:  # Classification task
#        return accuracy_score(y, model.predict(X))


def calculate_confusion_matrix(model, X, y, n_splits=5):
    y_pred = cross_val_predict(model, X, y, cv=n_splits)
    return confusion_matrix(y, y_pred)

def calculate_mean_absolute_error(model, X, y, n_splits=5):
    predictions = cross_val_predict(model, X, y, cv=n_splits)
    return mean_absolute_error(y, predictions)

def calculate_mean_squared_error(model, X, y, n_splits=5):
    predictions = cross_val_predict(model, X, y, cv=n_splits)
    return mean_squared_error(y, predictions)

def calculate_root_mean_squared_error(model, X, y, n_splits=5):
    predictions = cross_val_predict(model, X, y, cv=n_splits)
    mse = mean_squared_error(y, predictions)
    return np.sqrt(mse)

def calculate_precision(model, X, y, n_splits=5, avg_parameter='binary'):
    y_pred = cross_val_predict(model, X, y, cv=n_splits)
    return precision_score(y, y_pred, average=avg_parameter)

def calculate_recall(model, X, y, n_splits=5, avg_parameter='binary'):
    y_pred = cross_val_predict(model, X, y, cv=n_splits)
    return recall_score(y, y_pred, average=avg_parameter)

def calculate_f1_score(model, X, y, n_splits=5, avg_parameter='binary'):
    y_pred = cross_val_predict(model, X, y, cv=n_splits)
    return f1_score(y, y_pred, average=avg_parameter)


def calculate_silhouette_score(X, clustering_model):
    
    cluster_labels = clustering_model.fit_predict(X)
    
    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    return silhouette_avg

    

#def ks_drift_detection(data_new, historical_data_distribution=None, p_value=0.05):
#    """
#    Kolmogorov-Smirnov Test 
#    """
##    if historical_data_distribution is None:
##        # Calculate distribution from the original DataFrame
##        data_new = data_new.values.flatten() if isinstance(data_new, pd.DataFrame) else np.array(data_new)
##        mean = np.mean(data_new)
##        std_dev = np.std(data_new)
##        historical_data_distribution = np.random.normal(loc=mean, scale=std_dev, size=len(data_new))
#    # Convert input data to numpy arrays if they are pandas DataFrames or Series
#    if isinstance(data_new, (pd.DataFrame, pd.Series)):
#        data_new = data_new.values.flatten()
#    elif isinstance(data_new, list):
#        data_new = np.array(data_new)
#    if isinstance(historical_data_distribution, (pd.DataFrame, pd.Series)):
#        historical_data_distribution = historical_data_distribution.values.flatten()
#    elif isinstance(historical_data_distribution, list):
#        historical_data_distribution = np.array(historical_data_distribution)
#    # Perform the Kolmogorov-Smirnov test
#    ks_stat, p_val = stats.ks_2samp(data_new, historical_data_distribution)
#    # If the K-S statistic is small or the p-value is high (greater than the significance level, say 5%), then we cannot reject the hypothesis that the distributions of the two samples are the same.
#    if p_val < p_value: 
#        return 'Data Drift Detected'
#    else:
#        return 'No Data Drift'

def ks_drift_detection(data_new, data_compare, p_value=0.05):
    """
    Kolmogorov-Smirnov Test 
    """
    # Flatten input arrays
    data_new_flat = np.ravel(data_new)
    data_compare_flat = np.ravel(data_compare)
    
    # Perform the Kolmogorov-Smirnov test
    ks_stat, p_val = stats.ks_2samp(data_new_flat, data_compare_flat)

    # If the K-S statistic is small or the p-value is high (greater than the significance level, say 5%), then we cannot reject the hypothesis that the distributions of the two samples are the same.
    if p_val < p_value: 
        drift= 'Data Drift Detected, ks_drift:'
    else:
        drift= 'No Data Drift, ks_drift'

    #return ks_stat 
    return drift,ks_stat
#    if p_val < p_value: 
#        return 'Data Drift Detected'
#    else:
#        return 'No Data Drift'

def kl_divergence(data_compare, data_new):
    """
    Calculate Kullback-Leibler divergence
    """
    data_compare += 1e-10
    data_new += 1e-10
    kl_div = stats.entropy(data_compare, data_new)
    return "kl_divg: " + str(kl_div)

def js_divergence(data_compare, data_new):
    """
    Calculate Jensen-Shannon divergence
    """
    p = 0.5 * (data_compare + data_new)
    js_div = 0.5 * (stats.entropy(data_compare, p) + stats.entropy(data_new, p))
    return "js_divg: " + str(js_div)

