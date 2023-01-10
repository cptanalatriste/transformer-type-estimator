from base_analyser import TransformerTypeAnalyser
from prob_calibration import CalibratedTypeEstimator

SEED: int = 42
import numpy as np

np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)

import logging
from argparse import ArgumentParser, Namespace
from typing import List


def start_training(training_csv: str, testing_csv_file: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"training_csv {training_csv}")
    logging.info(f"testing_csv_file {testing_csv_file}")

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train_from_files(training_csv, testing_csv_file)


def obtain_probabilities(input_text: str, output_directory: str = "./model", calibration_method: str = "isotonic"):
    logging.basicConfig(level=logging.DEBUG)
    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser(output_directory=output_directory)
    calibrated_type_analyser: CalibratedTypeEstimator = CalibratedTypeEstimator(
        original_type_estimator=type_analyser, method=calibration_method,
        save_file=f"{output_directory}/{calibration_method}_calibrator.sav")

    prediction: List[float] = calibrated_type_analyser.obtain_probabilities(input_text)
    print(prediction[0])


def main():
    parser: ArgumentParser = ArgumentParser(
        description="A transformer-based intent recognition for answers to help requests.")
    parser.add_argument("--train_csv", type=str, help="CSV file with training data.")
    parser.add_argument("--test_csv", type=str, help="CSV file with testing data.")
    parser.add_argument("--input_text", type=str, help="Input text, for probability calculation.")
    parser.add_argument("--train", action="store_true", help="Start fine-tuning the pre-trained transformer model and "
                                                             "sending it to the hub.")
    parser.add_argument("--trainlocal", action="store_true", help="Start fine-tuning the pre-trained transformer model"
                                                                  " and saving it locally.")
    parser.add_argument("--pred", action="store_true", help="Calculate the probability for helping given a text"
                                                            " using a remote model.")
    parser.add_argument("--predlocal", action="store_true", help="Calculate the probability for helping given a text"
                                                                 " using a local model.")
    parser.add_argument("--modelDirectory", type=str, help="Directory where to load/store the "
                                                           "type estimator model.")
    arguments: Namespace = parser.parse_args()

    if arguments.train:
        start_training(arguments.train_csv, arguments.test_csv)
    elif arguments.trainlocal:
        start_training(arguments.train_csv, arguments.test_csv)
    elif arguments.pred:
        obtain_probabilities(arguments.input_text, output_directory=arguments.modelDirectory)
    elif arguments.predlocal:
        obtain_probabilities(arguments.input_text, output_directory=arguments.modelDirectory)


if __name__ == "__main__":
    main()
