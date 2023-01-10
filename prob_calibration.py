import logging
import pickle
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from base_analyser import TransformerTypeAnalyser


class CalibratedTypeEstimator:

    def __init__(self, original_type_estimator: TransformerTypeAnalyser, method: str,
                 save_file: Optional[str] = None):
        self.original_type_estimator: TransformerTypeAnalyser = original_type_estimator
        self.calibrator: Optional[Union[IsotonicRegression, LinearRegression]] = None
        if save_file is not None:
            self.calibrator = pickle.load(open(save_file, "rb"))
            logging.info(f"Calibrator loaded from {save_file}")

        self.method: str = method

    def calibrate(self, expressions: List[str], true_labels: List[int], number_of_bins: int) -> None:
        positive_probabilities: List[float] = self.original_type_estimator.obtain_probabilities(expressions,
                                                                                                local=True)
        bin_true_probabilities, bin_predicted_probabilities = calibration_curve(true_labels, positive_probabilities,
                                                                                n_bins=number_of_bins)

        if "isotonic" == self.method:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(bin_predicted_probabilities, bin_true_probabilities)
            logging.info("Isotonic Regression fitted.")
        elif "sigmoid" == self.method:
            bin_true_probabilities, bin_predicted_probabilities = filter_out_of_domain(bin_predicted_probabilities,
                                                                                       bin_true_probabilities)
            bin_true_probabilities: np.ndarray = np.log(bin_true_probabilities / (1 - bin_true_probabilities))
            self.calibrator = LinearRegression()
            self.calibrator.fit(bin_predicted_probabilities.reshape(-1, 1),
                                bin_true_probabilities.reshape(-1, 1))
            logging.info("Linear regression fitted.")

        save_file: str = f"model/{self.method}_calibrator.sav"
        pickle.dump(self.calibrator, open(save_file, "wb"))
        logging.info(f"Calibrator model saved at {save_file}")

    def obtain_probabilities(self, expressions: List[str]) -> List[float]:
        original_probabilities: List[float] = self.original_type_estimator.obtain_probabilities(expressions,
                                                                                                local=True)

        if "isotonic" == self.method:
            return self.calibrator.predict(original_probabilities)
        elif "sigmoid" == self.method:
            probabilities_as_array: np.ndarray = np.array(original_probabilities)
            return 1 / (1 + np.exp(-self.calibrator.predict(probabilities_as_array.reshape(-1, 1)).flatten()))


def filter_out_of_domain(predicted_probabilities, true_probabilities) -> np.ndarray:
    filtered = list(zip(*[probability
                          for probability in zip(predicted_probabilities, true_probabilities)
                          if 0 < probability[1] < 1]))
    return np.array(filtered)


def plot_reliability_diagram(true_labels: List[int], positive_probabilities: List[float], number_of_bins: int):
    bin_true_probabilities, bin_predicted_probabilities = calibration_curve(true_labels, positive_probabilities,
                                                                            n_bins=number_of_bins)
    error_value: float = expected_calibration_error(bin_true_probabilities,
                                                    bin_predicted_probabilities,
                                                    positive_probabilities)
    logging.info(
        f"Expected Calibration Error {error_value}")
    plt.hist(positive_probabilities,
             weights=np.ones_like(positive_probabilities) / len(positive_probabilities),
             alpha=.4, bins=np.maximum(10, number_of_bins))
    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")
    plt.plot(bin_predicted_probabilities, bin_true_probabilities, "s-", color="#162B37")
    plt.ylabel("Fraction of positives", )
    plt.xlabel("Mean predicted value")
    plt.legend()
    plt.grid(True, color="#B2C7D9")

    plt.show()


def expected_calibration_error(bin_true_probability: np.ndarray, bin_predicted_probability: np.ndarray,
                               person_type_probabilities: np.ndarray) -> float:
    number_of_bins: int = len(bin_true_probability)
    histogram = np.histogram(a=person_type_probabilities, range=(0, 1), bins=number_of_bins)
    bin_sizes = histogram[0]
    result: float = 0.0

    total_samples: float = float(sum(bin_sizes))
    for bin_index in np.arange(len(bin_sizes)):
        current_bin_size: int = bin_sizes[bin_index]
        true_probability: float = bin_true_probability[bin_index]
        predicted_probability: float = bin_predicted_probability[bin_index]

        result += current_bin_size / total_samples * np.abs(true_probability - predicted_probability)

    return result


def evaluate_type_analyser(types_in_data: List[int],
                           type_probabilities: List[float],
                           number_of_bins) -> pd.DataFrame:
    metrics: Dict[str, float] = {"brier_score_loss": brier_score_loss(types_in_data, type_probabilities),
                                 "log_loss": log_loss(types_in_data, type_probabilities),
                                 "roc_auc_score": roc_auc_score(types_in_data, type_probabilities)}

    bin_true_probabilities, bin_predicted_probabilities = calibration_curve(types_in_data, type_probabilities,
                                                                            n_bins=number_of_bins)
    metrics["expected_calibration_error"] = expected_calibration_error(bin_true_probabilities,
                                                                       bin_predicted_probabilities,
                                                                       type_probabilities)
    return pd.DataFrame(metrics.values(), index=metrics.keys())
