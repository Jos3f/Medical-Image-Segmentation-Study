import numpy as np
from numpy.linalg import norm
from skimage.filters import threshold_otsu
from pathos.multiprocessing import Pool

class FCM:
    """
    Fuzzy C-Means class.
    Currently using csFCM proposed by Adhikari et al. (2015) Conditional spatial fuzzy C-means clustering algorithm for segmentation of MRI images
    """

    """
    Constructor for the FCM class.

    Parameters
    ----------
    images : list
        A list of images (numpy arrays of (height, width, channels))
    num_clusters : int
        The number of clusters

    m : number
        Controls the fuzziness, default is 2

    p : number
        Importance of intensity information, default is 2
        N.b. a setting of p = 1, q = 0 corresponds to traditional FCM.

    q : number
        Importance of spatial information, default is 2
        N.b. a setting of p = 1, q = 0 corresponds to traditional FCM.

    epsilon : number
        Controls when convergence has been reached. The itertive process will 
        stop when the norm of each cluster center changes less than epsilon
        for any iteration. Default is 1e-5.

    window_size : int (odd)
        The size of the window used for spatial information. Larger values means 
        taking a larger (spatial) portion of the image into account for each 
        pixel. Default is 3.

    intialization: str
        Method used to initialize the cluster centers. Options are:
            'random': initialize randomly
            'otsu': Can only be used for binary segmentation (two clusters). 
                    Use Otsu thresholding (maximizing intra-class variance) to 
                    determine the intensity (per channel) which best separates 
                    the two clusters. The image is grouped into the pixels that 
                    exceed the threshold and those that do not. The initial 
                    centers are set to the average intensity of each group.
                    N.b. for multi-channel images this is done on a per-channel
                    basis and is likely to be unreliable.
    """
    def __init__(
        self,
        images, 
        num_clusters, 
        m = 2, p = 2, q = 2, 
        epislon = 1e-5, 
        window_size = 3, 
        initialization = 'random'):

        # Some sanity checks
        assert(isinstance(images, list))
        assert(window_size % 2 == 1) # Odd number

        self.__USE_OTSU_INITIALIZATION = False
        if initialization == 'otsu':
            self.__USE_OTSU_INITIALIZATION = True

        self.__MAX_ITERATIONS = 1000


        self.__images = images
        self.__num_clusters = num_clusters
        self.__m = m
        self.__p = p
        self.__q = q
        self.__epsilon = epislon
        self.__window_size = window_size

    def cluster(self):
        """
        Perform clustering on each image. 

        Returns
        -------
        A list of tuples (w, z), one for each image. w is the final cluster 
        centers, z are the fuzzy, weighted membership values for each pixel.
        """
        pool = Pool()
        return pool.map(self.__csFCM, self.__images)
        #return [self.__csFCM(image) for image in self.__images]

    def binary_segmentation(self):
        """
        Perform clustering on each image. 

        Returns
        -------
        A list of binary segmentations, one for each image.
        """
        clusterings = self.cluster()
        # Turn the pixels in low-intensity clusters to 0 and pixels in high-intensity 
        # cluster to 1
        segmentations = [
            self.__binary_segmentation_from_cluster_output(w, z)
            for (w, z) in clusterings
        ]
        return segmentations
        
    def __binary_segmentation_from_cluster_output(self, w, z):
        assert(w.shape[0] == 2) # With two clusters, this is fine
        if norm(w[0]) > norm(w[1]):
            return np.argmin(z, axis=0) 
        else:
            return np.argmax(z, axis=0) 

    def __csFCM(self, image):

        # Assuming first two axes are the image dimensions,
        # third is the shape of the data
        # TODO: No don't make weird (potentially unnecessary) coercions
        image_dimensions = image.shape[:2]
        image = image.reshape(*image_dimensions, -1)
        (image_height, image_width) = image_dimensions
        data_shape = image.shape[2:]
        
        # Normalize image
        image_min = image.min(axis=(0, 1), keepdims=True)
        image_max = image.max(axis=(0, 1), keepdims=True)
        normalized_image = (image - image_min) / (image_max - image_min) 
        image = normalized_image
        
        # Initial membership, doesn't need to be initialized
        u = np.zeros(shape=(self.__num_clusters, *image_dimensions))
        mu = np.zeros(shape=(self.__num_clusters, *image_dimensions))
        # Initial cluster centers
        # Initial joint cluster centers
        if self.__USE_OTSU_INITIALIZATION and self.__num_clusters == 2:
            # N.b. this is probably super sketchy for RGB images
            if data_shape[0] > 1:
                print('Warning: Using Otsu on each individual channel.')
            v, w = self.__otsu_initialization(image)
        else:
            if self.__USE_OTSU_INITIALIZATION:
                print('Warning: more than two clusters, Otsu impossible.')
            v = np.random.rand(self.__num_clusters, *data_shape)
            w = np.random.rand(self.__num_clusters, *data_shape)
        
        # Matrix to hold conditioning variable
        f = np.zeros(shape=(self.__num_clusters, *image_dimensions))
        # Matrix to hold weighted membership values
        z = np.zeros(shape=(self.__num_clusters, *image_dimensions))

        # View used when calculating the conditioning variable, 
        # (area around each pixel), only needs to be created once
        window = np.zeros(shape=(self.__num_clusters, *image_dimensions), dtype=object)
        step = int(self.__window_size / 2)
        for i in range(self.__num_clusters):
            for j in range(image_height):
                for k in range(image_width):
                    # n x n window with current pixel in center
                    # Determine if we're near the edge of the image
                    up = max(0, j - step)
                    down = min(j + step + 1, image_height)
                    left = max(0, k - step)
                    right = min(k + step + 1, image_width)
                    window[i][j][k] = mu[i, up:down, left:right]
        
        # Loop until done
        for counter in range(1, self.__MAX_ITERATIONS + 1):

            # This is useful
            center_distances = [
                    norm(image - v[i], axis=2)**(2/(self.__m - 1)) 
                    for i in range(self.__num_clusters)
            ]

            # Calculate membership value using cluster centers
            for i in range(self.__num_clusters):
                mu[i] = 1 / (center_distances[i] / sum(center_distances))

            # Determine f for each pixel
            for i in range(self.__num_clusters):
                for j in range(image_height):
                    for k in range(image_width):
                        f[i, j, k] = (
                            window[i, j, k].sum(axis=(0,1)) / 
                            window[i, j, k].size
                        )

            # Calculate conditional spatial membership value using cluster centers
            for i in range(self.__num_clusters):
                u[i] = f[i] / (center_distances[i] / sum(center_distances))

            # Calculate weighted membership value
            for i in range(self.__num_clusters):
                z[i] = (
                    (mu[i]**self.__p * u[i]**self.__q) / 
                    np.sum(mu**self.__p * u**self.__q, axis=0)
                )

            # Update joint cluster value 
            new_w = np.zeros(shape=w.shape)
            
            for i in range(self.__num_clusters):
                new_w[i] = (
                    np.sum((z[i]**self.__m)[:, :, None] * image, axis=(0, 1)) / 
                    np.sum(z[i]**self.__m)
                )

            # Update cluster centers
            for i in range(self.__num_clusters):
                v[i] = (
                    np.sum((mu[i]**self.__m)[:, :, None] * image, axis=(0, 1)) / 
                    np.sum(mu[i]**self.__m)
                )

            # Compare old and new w to see if we can end
            checkv = norm(new_w - w, axis=1) >= self.__epsilon
            
            w = new_w

            if np.any(checkv):
                continue
            else:
                # print('csFCM done after', counter, 'iterations!')
                # Output:
                #   w: joint cluster centers
                #   z: weighted membership per pixel
                return w, z

    def __otsu_initialization(self, image):

        data_shape = image.shape[2:]

        v = np.zeros(shape=(2, *data_shape))

        for channel in range(*data_shape):
            sub_image = image[:, :, channel]
            # Optimal threshold
            channel_threshold = threshold_otsu(sub_image)
            # Within-group average
            background_center = np.average(
                sub_image[sub_image <= channel_threshold]
            )
            foreground_center = np.average(
                sub_image[sub_image > channel_threshold]
            )
            v[0, channel] = background_center
            v[1, channel] = foreground_center

        return v, np.copy(v)
