import numpy as np
import matplotlib.pyplot as plt


class read_probability_masks():

    def __init__(self, directory):
        self.directory = directory

    def get_probability_mask(self, file_name):
        return np.load(self.directory + str("/") + file_name)

    def get_all_starting_w(self, name_starts_w):
        """
        Returns all masks from the files starting with a given string
        :param name_starts_w: the string that each file should start with
        :return: a list of the probability masks
        """
        return []

    def plot_mask(self, mask, tau=1):
        """

        :param mask:
        :param tau:
        :return:
        """
        plt.matshow(np.argmax(mask[0] * np.array([[[1, tau]]]), axis=-1), cmap=plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    # Example usage
    mask_reader = read_probability_masks("results")
    mask = mask_reader.get_probability_mask("BBBC039_test_fold_0.npy")
    print(mask)
    print(mask.shape)
    mask_reader.plot_mask(mask)


