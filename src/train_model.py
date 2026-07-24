import pickle


def save_model(model, filepath):
    """
    Saves a trained machine learning model.
    """

    with open(filepath, "wb") as file:
        pickle.dump(model, file)