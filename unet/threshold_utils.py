import numpy as np
from multiprocessing.dummy import Pool
from functools import partial
from metrics import Metrics

def normalize_output(output):
    """
    Turn the output for each pixel into a "probability" of the pixel being 
    white, for example:

        [0.5, 1.5] --> 0.75
    """
    return (output / np.sum(output, axis=2)[:, :, None])[:, :, 1]

def get_best_threshold(training_outputs, true_labels, min=1, max=10, num_steps=10, use_metric=0):
    """
    Calculates the best threshold (produces highest jaccard index)
    :param training_outputs:
    :param num_steps:
    :return:
    """
    best_tau = 1
    best_score = 0
    if use_metric == 3:
        best_score = 1

    thresholds = [min + i * (max-min) / num_steps for i in range(num_steps)]
    print("Evaluating", num_steps, "thresholds...", end=' ', flush=True)

    f = partial(score_from_t, true_labels, training_outputs, min, max, num_steps, use_metric)
    with Pool(processes=4) as pool:
        scores = np.array(pool.map(f, thresholds))

        if use_metric == 3:
            idx = np.argmin(scores)
        else:
            idx = np.argmax(scores)
        print("Done")

        best_score = scores[idx]
        best_tau = thresholds[idx]
    
    return best_tau, best_score

def score_from_t(true_labels, training_outputs, start, stop, num_steps, use_metric, tau):
    #print("Calculating for threshold", tau)
    step = int((num_steps * tau - start) / (stop-start))
    #print("\rEvaluating threshold", step, "/", num_steps, end='', flush=True)
    print(tau, end=',', flush=True)

    temp_predictions = []
    #print("Appending")
    for j in range(len(training_outputs)):
        temp_predictions.append((training_outputs[j] >= tau).astype(int))
    #print("Creating metric object")
    metric = Metrics(true_labels, temp_predictions, safe=False, parallel=False)

    #print("Running metric function")
    if use_metric == 1:
        scores = metric.adj_rand()
    elif use_metric == 3:
        scores = metric.warping_error()
    else:
        scores = metric.jaccard()

    avg_score = sum(scores) / len(scores)
    #print("Score for", tau, "-", avg_score)
    return avg_score