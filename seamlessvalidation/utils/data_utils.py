import pickle


def generate_test_datadriftdata (sample_size=1000, num_features=1, drift_fraction=0.1):
    data_old_mean = np.random.uniform(0, 10, num_features)
    data_old_std = np.random.uniform(1, 3, num_features)
    data_old = np.random.normal(data_old_mean, data_old_std, size=(sample_size, num_features))
    drift_indices = np.random.choice(sample_size, int(sample_size * drift_fraction), replace=False)
    data_new = np.copy(data_old)
    for feature_idx in range(num_features):
        data_new[drift_indices, feature_idx] += np.random.uniform(5, 10)  # Adding drift

    return data_old, data_new

data_compare,data_new=generate_test_datadriftdata()




def split_data(data, target_column, test_size=0.2, random_state=None):
    """
    Split data into training and testing sets.

    Parameters:
    - data: DataFrame containing the data.
    - target_column: Name of the target column.
    - test_size: Size of the testing set (default is 0.2).
    - random_state: Random seed for reproducibility (default is None).

    Returns:
    - Tuple containing X_train, X_test, y_train, y_test.
    """
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def imputation(data):
    """
    Impute missing values.

    Parameters:
    - data: DataFrame containing the data.

    Returns:
    - DataFrame with missing values handled.
    """
    # Handle missing values with mean imputation
    return data.fillna(data.mean())


def serialize_data(data, file_path):
    """
    Serialize data and save it to a file.

    Parameters:
    - data: Data object to be serialized.
    - file_path: Path to save the serialized data.

    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def deserialize_data(file_path):
    """
    Deserialize data from a file.

    Parameters:
    - file_path: Path to the serialized data file.

    Returns:
    - Deserialized data object.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data