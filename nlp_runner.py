import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from base_analyser import TransformerTypeAnalyser

SEED: int = 0


def predict(tweet_to_predict: str):
    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    prediction: float = type_analyser.obtain_probabilities(tweet_to_predict, local=True)
    print(tweet_to_predict, prediction)


def preprocess_data(original_data_file: str, target_data_file: str, total_agreement: bool = True) -> None:
    original_dataframe: pd.DataFrame = pd.read_excel(original_data_file, engine='openpyxl')
    logging.info(f"Original dataframe: {len(original_dataframe.index)} rows")

    if total_agreement:
        original_dataframe = original_dataframe.loc[original_dataframe["Marker 1"] == original_dataframe["Marker 2"]]
        logging.info(f"After agreement filtering: {len(original_dataframe.index)} rows")

    original_dataframe["will_help"] = original_dataframe["Marker 1"] == "Group"
    original_dataframe = original_dataframe.drop(["Marker 1", "Marker 2", "Marker 3"], axis=1)

    original_dataframe.to_csv(target_data_file, index=False)
    logging.info(f"Pre-processed data file at {target_data_file}")


def load_and_split(data_file: str, test_size: float = 0.5) -> Tuple[str, str]:
    original_dataframe: pd.DataFrame = pd.read_csv(data_file)
    original_dataframe["will_help"] = original_dataframe["will_help"].astype(int)

    training_dataframe: pd.DataFrame
    testing_dataframe: pd.DataFrame
    training_csv_file: str = "data/training_data.csv"
    testing_csv_file: str = "data/testing_data.csv"

    training_dataframe, testing_dataframe = train_test_split(original_dataframe, test_size=test_size)
    logging.info(
        f"{len(training_dataframe.index)} training instances, {len(testing_dataframe.index)} testing instances")

    training_dataframe.to_csv(training_csv_file, index=False)
    testing_dataframe.to_csv(testing_csv_file, index=False)

    return training_csv_file, testing_csv_file


def train(data_file: str = "data/survivor_responses.csv"):
    training_csv_file, testing_csv_file = load_and_split(data_file)

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train_from_files(training_csv_file, testing_csv_file)


def main():
    # train()

    target_data_file: str = "data/survivor_responses.csv"
    # preprocess_data(original_data_file="data/final data_agreed.xlsx",
    #                 target_data_file=target_data_file)
    load_and_split(data_file=target_data_file, test_size=0.3)
    # predict("Sure! I'll help")
    # predict("Sorry, I can't")


if __name__ == "__main__":
    np.random.seed(SEED)

    logging.basicConfig(level=logging.INFO)
    main()
