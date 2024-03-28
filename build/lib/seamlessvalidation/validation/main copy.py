from seamlessvalidation import pd
from seamlessvalidation import np
from seamlessvalidation import plt
from seamlessvalidation import io




from sklearn import metrics

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import make_classification, make_regression

from sklearn.base import ClassifierMixin, RegressorMixin,BaseEstimator, ClusterMixin

from sklearn.model_selection import (cross_val_score, cross_val_predict, StratifiedKFold,LeaveOneOut, KFold)

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_curve,
                             auc, mean_absolute_error, mean_squared_error, roc_auc_score,
                             f1_score,recall_score,precision_score,silhouette_score)


# Function to check if the model is a clustering algorithm
def is_clustering_model(model):
    if not issubclass(model.__class__, BaseEstimator):
        return False
    if issubclass(model.__class__, ClusterMixin) or hasattr(model, "fit_predict"):
        return True
    return False

def calculate_silhouette_score(X, clustering_model):
    
    cluster_labels = clustering_model.fit_predict(X)
    
    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    return silhouette_avg


def cross_validation(model, X, y, n_splits=5, cv_strategy='k_fold',avg_strategy='weighted'):
    # Determine the cross-validation strategy
    if cv_strategy == 'stratified_k_fold' and isinstance(model, ClassifierMixin):
        cv = StratifiedKFold(n_splits=n_splits)
    elif cv_strategy == 'leave_one_out':
        cv = LeaveOneOut()
    elif cv_strategy == 'k_fold':
        cv = KFold(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits)

    if avg_strategy == 'weighted':
        avg_parameter= 'weighted'
    elif avg_strategy == 'micro':
        avg_parameter = 'micro'
    elif avg_strategy == 'macro':
        avg_parameter = 'macro'
    elif avg_strategy == 'samples':
        avg_parameter = 'samples'
    else:
        avg_parameter ='binary'
        
    # Initialize metrics
    mean_accuracy_or_r2 = None
    mae = mse = rmse = roc_auc =precision=recall=f1= precision_recall_auc = None
    cm = pr_curve = None

    # Check if the model is a classifier or a regressor
    is_classifier = isinstance(model, ClassifierMixin)
    is_regressor = isinstance(model, RegressorMixin)
    
    scoring_method = 'accuracy' if is_classifier else 'r2'
    mean_accuracy_or_r2 = cross_val_score(model, X, y, cv=cv, scoring=scoring_method).mean()

    # For classifiers, calculate accuracy and possibly ROC-AUC, and confusion matrix
    if is_classifier:
        # Calculate mean cross-validated accuracy
        mean_accuracy_or_r2 = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy').mean()

        # Calculate confusion matrix
        y_pred = cross_val_predict(model, X, y, cv=n_splits)
        cm = confusion_matrix(y, y_pred)

    
        # Calculate ROC-AUC for binary classification if model supports probability predictions
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
                #x=plt.show()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)  # Move to the start of the BytesIO object

    # For regressors, calculate R^2
    elif is_regressor:
        mean_accuracy_or_r2 = cross_val_score(model, X, y, cv=n_splits, scoring='r2').mean()

    # Calculate MAE, MSE, and RMSE for both classifiers and regressors
    predictions = cross_val_predict(model, X, y, cv=n_splits)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    y_true = y
    precision = precision_score(y_true, y_pred, average=avg_parameter)
    recall = recall_score(y_true, y_pred, average=avg_parameter)
    f1 = f1_score(y_true, y_pred, average=avg_parameter)
        
    # Compile results
    metrics_dict = {
        'Strategy' : cv,
        'Mean Accuracy (Classifiers) or R^2 (Regressors)': mean_accuracy_or_r2,
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'ROC-AUC (Binary Classifiers only)': roc_auc,
        'Confusion Matrix (Classifiers only)': cm,
        #'Precision-Recall Curve (Binary Classifiers only)': pr_curve,
        #'plot1': buf,
        'Precision': precision,
        'Recall': recall,
        'f1score':f1,
        'Precision-Recall AUC (Binary Classifiers only)': precision_recall_auc,
        #'silhouette score':silhouette_score
    }
    return metrics_dict



def seamlessvalidation(model, **kwargs):
    X=kwargs.get('X_class', None)
    y=kwargs.get('y_class', None)
    n_splits=kwargs.get('n_splits', None)
    cv_strategy=kwargs.get('cv_strategy', None)
    is_classifier = isinstance(model, ClassifierMixin)
    is_regressor = isinstance(model, RegressorMixin)
    is_clustering= is_clustering_model(model)
    if is_clustering:
        metric=calculate_silhouette_score(X, model)
        metrics_dict = {
        'silhouette score':metric}
    else:
        metrics_dict=cross_validation(model, X, y, n_splits, cv_strategy)
    return metrics_dict
        