from src.data_loader import load_dataset
from src.data_loader import dataset_summary


DATA_PATH = "data/IMDB Dataset.csv"


def main():
    print("\nStarting Sentiment Analysis Project...\n")

    # Load dataset
    df = load_dataset(DATA_PATH)

    # Display dataset information
    dataset_summary(df)


if __name__ == "__main__":
    main()