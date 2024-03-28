from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, DBSCAN

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
