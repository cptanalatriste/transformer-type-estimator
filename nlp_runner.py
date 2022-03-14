import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformer_analyser import TransformerTypeAnalyser

SEED: int = 0


def predict(tweet_to_predict: str):
    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    prediction: float = type_analyser.obtain_probabilities(tweet_to_predict)
    print(tweet_to_predict, prediction)


def train(csv_file: str = "data/survivor_responses.csv"):
    np.random.seed(SEED)

    original_dataframe: pd.DataFrame = pd.read_csv(csv_file)
    original_dataframe["will_help"] = original_dataframe["will_help"].astype(int)

    training_dataframe: pd.DataFrame
    testing_dataframe: pd.DataFrame
    test_size: float = 0.5
    training_csv_file: str = "training_data.csv"
    testing_csv_file: str = "testing_data.csv"

    training_dataframe, testing_dataframe = train_test_split(original_dataframe, test_size=test_size)
    training_dataframe.to_csv(training_csv_file, index=False)
    testing_dataframe.to_csv(testing_csv_file, index=False)

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train(training_csv_file, testing_csv_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # train()
    predict("Sure! I'll help")
    predict("Sorry, I can't")
