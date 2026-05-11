def display_basic_info(data):
    """
    Display basic information about the dataset.
    """
    print("Dataset Information:")
    print(data.info())

    print("\nDataset Description:")
    print(data.describe())

    print("\nNumber of Unique Values:")
    print(data.nunique())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nFirst 5 Rows:")
    print(data.head())
