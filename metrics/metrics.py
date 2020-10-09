import numpy as np
import sklearn.metrics
import random
from skimage import measure
from scipy.ndimage.morphology import distance_transform_edt
import timeit

class Metrics:
    """
    Various segmentation metrics on binary images.
    Currently implemented:

        Jaccard index (IoU)
        Dice score (also known as F1)
        Adjusted rand index
        Warping error (Jain et al. (2010)  Boundary Learning by Optimization with Topological Constraints).
    """
    def __init__(self, true_labels, inferred_labels):
        """
        Constructor for the Metrics class.

        Parameters
        ----------
        true_labels : list
            A list labelings, each labeling corresponding to one image.
            Each labeling is a binary 2D numpy array. With 1 for foreground
            and 0 for background.
        inferred_labels : str
            The output from the segmentation model. Same format as true_labels.
        """

        # Try to parse input
        if not isinstance(true_labels, list):
            true_labels = [true_labels]
        
        if not isinstance(inferred_labels, list):
            inferred_labels = [inferred_labels]

        
        # Assert equal number of images
        assert(len(true_labels) == len(inferred_labels))
        # For all pairs (true, inferred), assert the following:
        #   1. Both have identical shape
        #   2. Both are binary 
        assert(all(
            a.shape == b.shape and # 1
            np.array_equal(a, a.astype(bool)) and # 2
            np.array_equal(b, b.astype(bool)) # 2
            for (a, b) in zip(true_labels, inferred_labels)
        ))

        self.true_labels = true_labels
        self.inferred_labels = inferred_labels

        ### TESTING: TODO
        self.__topological_map = None
        ### TESTING DONE

    def jaccard(self):
        """
        Calculate the Jaccard index of every segmentation. 
        The scores are between 0 and 1. Higher values are better.

        Returns
        -------
        A list of Jaccard scores, one for each image. The index of the 
        score in the list corresponds to the index of the segmentation
        in true_labels and inferred_labels.
        """
        return self.__apply_metric_on_all_images(self.__jaccard_for_image)

    def dice(self):
        """
        Calculate the dice score of every segmentation.
        The scores are between 0 and 1. Higher values are better.

        Returns
        -------
        A list of dice scores, one for each image. The index of the 
        score in the list corresponds to the index of the segmentation
        in true_labels and inferred_labels.
        """
        return self.__apply_metric_on_all_images(self.__dice_for_image)

    def adj_rand(self):
        """
        Calculate the adjusted rand index of every segmentation.
        The scores are between 0 and 1. Higher values are better.

        Returns
        -------
        A list of adjusted rand scores, one for each image. The index of the 
        score in the list corresponds to the index of the segmentation
        in true_labels and inferred_labels.
        """
        return self.__apply_metric_on_all_images(self.__rand_for_image)

    def warping_error(self):
        """
        Calculate the warping error of every segmentation. 
        The scores are between 0 and 1. Note that the warping error is defined 
        as the Hamming distance between the inferred labels and the most 
        similar warping of the true labels. Therefore, lower values are better.

        Returns
        -------
        A list of warping errors, one for each image. The index of the 
        score in the list corresponds to the index of the segmentation
        in true_labels and inferred_labels.
        """
        return self.__apply_metric_on_all_images(
            lambda x, y: self.__warping_error_for_image(x, y)['distance']
        )

    def __apply_metric_on_all_images(self, metric_function):
        return [
            metric_function(x, y) 
            for (x, y) 
            in zip(self.true_labels, self.inferred_labels)
        ]

    def __dice_for_image(self, true_mask, inferred_mask):
        return sklearn.metrics.f1_score(true_mask.reshape(-1), inferred_mask.reshape(-1))

    def __jaccard_for_image(self, true_mask, inferred_mask):
        count_intersection = np.count_nonzero((inferred_mask == 1) & (true_mask == 1))
        count_union = (
            np.count_nonzero(inferred_mask) + 
            np.count_nonzero(true_mask) - 
            count_intersection
        )
        return count_intersection / count_union
        # Equivalently:
        # return sklearn.metrics.jaccard_score(true_mask.reshape(-1), inferred_mask.reshape(-1))

    # Rand score
    def __rand_for_image(self, true_mask, inferred_mask):
        return sklearn.metrics.adjusted_rand_score(true_mask.reshape(-1), inferred_mask.reshape(-1))

    # Get all pixels within a euclidean distance of n from the background
    # n controls how much the foreground is allowed to shrink (while preserving topology)
    def __generate_mask(self, image, n = 4):
        return (distance_transform_edt(image) <= n)
        
    def __topological_numbers_are_one(self, warped_labels, pixel_index):
        (i, j) = pixel_index
        fg_window = warped_labels[
            max(i - 1, 0):min(i + 2, warped_labels.shape[0]),
            max(j - 1, 0):min(j + 2, warped_labels.shape[1])
        ]

        bg_window = warped_labels[
            max(i - 1, 0):min(i + 2, warped_labels.shape[0]),
            max(j - 1, 0):min(j + 2, warped_labels.shape[1])
        ]
        
        ### TESTING. TODO: MORE OPTIMIZATION
        if fg_window.shape == (3, 3):
            hash_val = self.__hash_binary_array(fg_window)
            return self.__topological_map[hash_val]
        ### TESTING DONE

        # Adjust fg window for 4-adjacency
        if fg_window.shape == (2, 2):
            # Corner
            if i == 0 and j == 0:
                mask = np.array([   
                    [1, 1], 
                    [1, 0]
                ])
            elif i == 0 and j > 0:
                mask = np.array([
                    [1, 1],
                    [0, 1]
                ])
            elif i > 0 and j == 0:
                mask = np.array([
                    [1, 0], 
                    [1, 1],
                ])
            else: # i > 0 and j > 0
                mask = np.array([
                    [0, 1], 
                    [1, 1],
                ])

        elif fg_window.shape == (3, 2):
            # Along a vertical edge, so either j = 0 or j = width - 1
            if j == 0:
                mask = np.array([
                    [1, 0], 
                    [1, 1],
                    [1, 0]
                ])
            else:
                mask = np.array([
                    [0, 1], 
                    [1, 1],
                    [0, 1]
                ])

        elif fg_window.shape == (2, 3):
            # Along a horizontal edge, so either i = 0 or i = height - 1
            if i == 0:
                mask = np.array([
                    [1, 1, 1], 
                    [0, 1, 0]
                ])
            else:
                mask = np.array([
                    [0, 1, 0], 
                    [1, 1, 1]
                ])
        else:
            # Normal 3x3 window
            mask = np.array([
                [0, 1, 0], 
                [1, 1, 1],
                [0, 1, 0]
            ])


        # Backup middle element
        old_val = warped_labels[pixel_index]
        
        # Mask middle element
        warped_labels[pixel_index] = 0
        # Zeroing out components not in the 4-neighborhood through element-wise
        # multiplication with binary mask
        fg_number = np.count_nonzero(np.unique(
            measure.label(fg_window, connectivity=1, background=0) * mask
        ))

        # Mask middle element
        warped_labels[pixel_index] = 1
        bg_number = np.count_nonzero(np.unique(
            measure.label(bg_window, connectivity=2, background=1)
        ))

        # Restore value
        warped_labels[pixel_index] = old_val

        return ((fg_number == 1) and (bg_number == 1))

    def __update_topological_info(self, true_labels, simple, pixel_index):
        for i in range(
            max(pixel_index[0] - 1, 0), 
            min(pixel_index[0] + 2, simple.shape[0])
        ):
            for j in range(
                max(pixel_index[1] - 1, 0), 
                min(pixel_index[1] + 2, simple.shape[1])
            ):
                simple[i, j] = self.__topological_numbers_are_one(true_labels, (i, j))
        # Done updating topoligcal information

    ### TESTING
    def __create_map(self):
        topological_map = np.zeros(512, dtype=bool)
        self.__hash_dot = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
        def binary_matrices(n = 3):
            shift = np.arange(n*n).reshape(n, n)
            for j in range(2**(n*n)):
                yield j >> shift & 1

        for m in binary_matrices():
            #hash_val = str(m)
            hash_val = self.__hash_binary_array(m)
                # Normal 3x3 window
            mask = np.array([
                [0, 1, 0], 
                [1, 1, 1],
                [0, 1, 0]
            ])

            # Backup middle element
            old_val = m[1, 1]
            
            # Mask middle element
            m[1, 1] = 0
            # Zeroing out components not in the 4-neighborhood through element-wise
            # multiplication with binary mask
            fg_number = np.count_nonzero(np.unique(
                measure.label(m, connectivity=1, background=0) * mask
            ))

            # Mask middle element
            m[1, 1] = 1
            bg_number = np.count_nonzero(np.unique(
                measure.label(m, connectivity=2, background=1)
            ))

            topological_map[hash_val] = ((fg_number == 1) and (bg_number == 1))
        return topological_map
    ### TESTING DONE

    ### TESTING
    def __hash_binary_array(self, a):
        return np.dot(a.reshape(-1), self.__hash_dot)
    ### TESTING DONE

    def __warping_error_for_image(  self,
                                    true_mask, 
                                    inferred_labels, 
                                    record_history=[], 
                                    return_final_warping=False):

        return_dict = {}
        history = {}
        for key in record_history:
            history[key] = []

        RECORD_LABELS = False
        RECORD_DISTANCE = False
        RECORD_FLIPPABLE = False
        RECORD_FLIPPABLE_COUNT = False


        if 'labels' in record_history:
            RECORD_LABELS = True
        if 'distance' in record_history:
            RECORD_DISTANCE = True
        if 'flippable' in record_history:
            RECORD_FLIPPABLE = True
        if 'flippable_count' in record_history:
            RECORD_FLIPPABLE_COUNT = True

        # Since we're modifying the true mask, we better copy it
        warped_labels = np.copy(true_mask)

        # Initialize variables
        # Mask of pixels that are close enough to the background 
        # that they are allowed to flip
        mask = self.__generate_mask(warped_labels)

        # To determine the Hamming distance, we need the number of 
        # pixels that are different, and the total number of pixels 
        different = np.logical_xor(warped_labels, inferred_labels)
        n = warped_labels.shape[0] * warped_labels.shape[1] # height x width
        
        # Simple points are those that have the topological numbers 1 for both
        # foreground and background 
        simple = np.zeros(
            shape=(warped_labels.shape[0], warped_labels.shape[1]),
            dtype=bool
        )

        ### TESTING. TODO: More optimization
        if not np.any(self.__topological_map):
            self.__topological_map = self.__create_map()
        ### TESTING DONE

        for i in range(warped_labels.shape[0]):
            for j in range(warped_labels.shape[1]):
                simple[i, j] = self.__topological_numbers_are_one(warped_labels, (i, j))


        # Only the points that are simple and in the mask are allowed to flip,
        # additionally, in a binary image only the points that are different 
        # will decrease Hamming distance
        flippable = (simple & mask & different)
        start = timeit.default_timer()
        flips = 0
        while True:
            if RECORD_DISTANCE:
                hamming_distance = np.count_nonzero(different) / n
                history['distance'].append(hamming_distance)
        
            if RECORD_FLIPPABLE:
                history['flippable'].append(np.copy(flippable))

            if RECORD_FLIPPABLE_COUNT:
                history['flippable_count'].append(np.count_nonzero(flippable))
            
            if RECORD_LABELS:
                history['labels'].append(np.copy(warped_labels))

            # Pick a random index out of pixels that are allowed to flip and 
            # decrease Hamming distance (break ties randomly)
            idxs = np.nonzero(flippable)
            if (len(idxs[0]) == 0):
                # No more points are flippable, local minimum
                #print("Done after", flips, "flips!")
                break
            x = np.random.randint(0, len(idxs[0]))
            pixel_index = (idxs[0][x], idxs[1][x])

            # Flip this pixel
            # The xor is safe as long as the image is binary
            warped_labels[pixel_index] = warped_labels[pixel_index] ^ 1
            
            # These pixels used to be different, so they obviously aren't now
            different[pixel_index] = False

            # Update information about which points are simple in the 
            # neighborhood of the recently changed pixel.
            self.__update_topological_info(warped_labels, simple, pixel_index)
            
            # Update array of flippable pixels
            s1 = slice(max(0, pixel_index[0] - 1), min(flippable.shape[0], pixel_index[0] + 2))
            s2 = slice(max(0, pixel_index[1] - 1), min(flippable.shape[1], pixel_index[1] + 2))
            flippable[s1, s2] = (
                simple[s1, s2] & different[s1, s2] & mask[s1, s2]
            )
            
            flips += 1

        different = np.logical_xor(warped_labels, inferred_labels)
        hamming_distance = np.count_nonzero(different) / n
        
        #print("Iterative process took", timeit.default_timer() - start, "total flips:", flips)
        return_dict['distance'] = hamming_distance

        if len(record_history) > 0:
            return_dict['history'] = history
        if return_final_warping:
            return_dict['final_warping'] = warped_labels

        return return_dict