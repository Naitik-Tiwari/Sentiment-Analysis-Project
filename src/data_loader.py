import pandas as pd


def load_dataset(filepath):
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath)
    return df


def dataset_summary(df):
    """
    Display important information about the dataset.
    """

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nNumber of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nDataset Info")
    print("-" * 60)
    df.info()

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nSentiment Distribution")
    print(df["sentiment"].value_counts())

    print("\nFirst Five Rows")
    print(df.head())

    print("\nLast Five Rows")
    print(df.tail())