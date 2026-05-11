from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_features_target(data, target_column):
    """
    Separate the dataset into features (X) and target variable (y).
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def scale_features(X):
    """
    Scale feature values using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
