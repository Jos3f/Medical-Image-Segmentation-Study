from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_for_qualitative_eval(image, ground_truth, prediction, data_point_index=0):
    """Only tested for 039"""
    fig, ax = plt.subplots(1, 2, figsize=(34, 22))
    fig.suptitle("Test point: " + str(data_point_index), fontsize=72)
    # Custom cmap
    custom_map = plt.cm.gray(np.linspace(0, 1, 256))
    true_color = np.array([0, 1, 0, 1])
    predicted_color = np.array([1, 0, 0, 1])
    empty_color = np.array([0, 0, 0, 0])
    map_width = custom_map.shape[0]
    custom_map[int(map_width * 0.2): int(map_width * 0.4)] = true_color
    custom_map[int(map_width * 0.4): int(map_width * 0.8)] = predicted_color
    custom_map_0 = ListedColormap(custom_map)

    ax[0].matshow(image, cmap=plt.cm.inferno)
    ax[0].set_title("Input data", fontsize=48)
    ax[0].set_axis_off()

    union_mask = ground_truth * prediction
    true_mask = (ground_truth - union_mask) * 0.3
    pred_mask = (ground_truth - union_mask) * 0.6
    ax[1].matshow(union_mask + true_mask + pred_mask, cmap=custom_map_0)
    ax[1].set_title("True and predicted mask", fontsize=48)
    ax[1].set_axis_off()

    fig.tight_layout()
    # plt.savefig("results/BBBC039/fig_" + str(data_point_index) + "combined.png")
    plt.show()


def plot_scilife(original_images, metric_predictions):
    """
    Takes 5 images and plots them
    :param original_images: list of five images
    :param metric_predictions: List of 5 predictions
    :return:
    """
    now = datetime.now()
    current_dt = now.strftime("%y_%m_%d_%H_%M_%S")
    fig, ax = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(25, 60))

    for i in range(5):
        ax[i][0].matshow(original_images[i])
        ax[i][1].matshow(metric_predictions[i], cmap=plt.cm.gray)

    plt.tight_layout()
    plt.savefig("results/scilifelab_" + str(current_dt) + ".png")
    plt.show()
    pass

if __name__ == '__main__':
    pass