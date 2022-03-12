import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from analyser import TransformerTypeAnalyser

SEED: int = 0


def main(csv_file: str = "data/survivor_responses.csv"):
    np.random.seed(SEED)

    original_dataframe: pd.DataFrame = pd.read_csv(csv_file)
    training_dataframe: pd.DataFrame
    testing_dataframe: pd.DataFrame
    test_size: float = 0.5
    training_csv_file: str = "training_data.csv"
    testing_csv_file: str = "testing_data.csv"

    training_dataframe, testing_dataframe = train_test_split(original_dataframe, test_size=0.5)
    training_dataframe.to_csv(training_csv_file, index=False)
    testing_dataframe.to_csv(testing_csv_file, index=False)

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train(training_csv_file, testing_csv_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
