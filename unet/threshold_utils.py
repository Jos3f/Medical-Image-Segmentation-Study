import numpy as np
from metrics import Metrics

class Threshold:


    def __init__(self, true_labels):
        self.true_labels = true_labels

    def get_best_threshold(self, training_outputs, min=1, max=10, num_steps=10, metric=0):
        """
        Calculates the best threshold (produces highest jaccard index)
        :param training_outputs:
        :param num_steps:
        :return:
        """

        best_tau = 1
        best_score = 0
        if metric == 3:
            best_score = 1

        metric = Metrics(self.true_labels, self.true_labels)

        for i in range(0, num_steps):
            tau = min + i * 1.0 * (max-min) / num_steps
            temp_predictions = []
            for j in range(len(training_outputs)):
                temp_predictions.append(training_outputs[j] * np.array([[[1, tau]]]))
                temp_predictions[j] = np.argmax(temp_predictions[j], axis=-1)

            metric.inferred_labels = temp_predictions


            if metric == 1:
                scores = metric.dice()
            elif metric == 3:
                scores = metric.warping_error()
            else:
                scores = metric.jaccard()

            avg_score = sum(scores) * 1.0 / len(scores)

            if (avg_score > best_score and metric != 3) or (avg_score < best_score and metric == 3):
                best_score = avg_score
                best_tau = tau

        return best_tau, best_score
