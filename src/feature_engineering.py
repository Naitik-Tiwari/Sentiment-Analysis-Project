from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_vectorizer():
    """
    Creates a TF-IDF vectorizer.
    """

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8
    )

    return vectorizer

def fit_transform_reviews(vectorizer, reviews):
    """
    Learns the vocabulary and converts reviews into TF-IDF vectors.
    """

    X = vectorizer.fit_transform(reviews)

    return X

def save_vectorizer(vectorizer, filepath):
    """
    Saves the trained TF-IDF vectorizer.
    """

    with open(filepath, "wb") as file:
        pickle.dump(vectorizer, file)