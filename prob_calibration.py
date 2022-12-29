import logging
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from transformer_analyser import TransformerTypeAnalyser


class CalibratedTypeEstimator:

    def __init__(self, original_type_estimator: TransformerTypeAnalyser):
        self.original_type_estimator: TransformerTypeAnalyser = original_type_estimator
        self.isotonic_calibrator: Optional[IsotonicRegression] = None

    def calibrate(self, expressions: List[str], true_labels: List[int], number_of_bins: int) -> None:
        positive_probabilities: List[float] = self.original_type_estimator.obtain_probabilities(expressions,
                                                                                                local=True)
        bin_true_probabilities, bin_predicted_probabilities = calibration_curve(true_labels, positive_probabilities,
                                                                                n_bins=number_of_bins)

        self.isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
        self.isotonic_calibrator.fit(bin_predicted_probabilities, bin_true_probabilities)
        logging.info("Isotonic Regression fitted.")

    def obtain_probabilities(self, expressions: List[str]) -> List[float]:
        original_probabilities: List[float] = self.original_type_estimator.obtain_probabilities(expressions,
                                                                                                local=True)
        return self.isotonic_calibrator.predict(original_probabilities)


def plot_reliability_diagram(true_labels: List[int], positive_probabilities: List[float], number_of_bins: int):
    bin_true_probabilities, bin_predicted_probabilities = calibration_curve(true_labels, positive_probabilities,
                                                                            n_bins=number_of_bins)
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
