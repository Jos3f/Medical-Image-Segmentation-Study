import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import circle
import math

class DataGenerator:

    def __init__(self, dims, seed=1337):
        self.dims = dims
        np.random.seed(seed)
        self.target = np.array([[0, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 1, 0]])
        self.target = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

        self.target = np.ones((24, 24)).astype(int)

    def sample_images(self, n=100, targets=300, bound = 0.6, ratio = 0.5, image_dir=None, label_dir=None):
        """
        ratio: What ratio of the added circles have low intensity and are not to included in the label

        bound: the bound between the values high and low intensity
        """


        if image_dir is not None:
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
        if label_dir is not None:
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

        images = []
        labels = []
        for i in range(n):
            image, label = self.sample_image(bound,ratio,targets)
            # images.append(image)
            # labels.append(label)
            if image_dir is not None:
                im = Image.fromarray(image)
                im.save(image_dir + '/' + str(i) + ".tif")
            if label_dir is not None:
                im = Image.fromarray(label)
                im.save(label_dir + '/' + str(i) + ".tif")

        return images, labels



        pass

    def sample_image(self, bound, ratio, targets=300):
        generated_label = np.zeros(self.dims).astype(int)
        generated_image = np.zeros(self.dims).astype(float)
        positions = self._get_random_coordinates(targets, margin=(self.target.shape[0]))
        radius=10
        for i in range(int(math.floor(targets * ratio))):
            x = positions[i][0]
            y = positions[i][1]
            rr,cc = circle(x,y,radius,generated_label.shape)

            generated_label[rr,cc] = 255
            generated_image[rr, cc] = np.random.uniform(bound, 1)


        for i in range(int(math.floor(targets * ratio)),targets):
            x = positions[i][0]
            y = positions[i][1]
            rr,cc = circle(x,y,radius,generated_label.shape)
            generated_image[rr, cc] = np.random.uniform(0.15, bound)

        generated_label = generated_label.clip(max=255)

        # generated_image = gaussian_filter(generated_image, sigma=2)
        generated_image = gaussian_filter(generated_image, sigma=10)
        generated_image = generated_image + np.random.normal(0.005, 0.01, generated_image.shape)
        generated_image = generated_image.clip(min=0, max=1)

        # generated_image = gaussian_filter(generated_label, sigma=2)
        generated_image = (generated_image / np.max(generated_image) * 255).astype(int)

        # plt.rcParams["figure.figsize"] = (20, 20)
        # plt.tight_layout()
        # plt.matshow(generated_label)
        # plt.show()
        #
        # plt.matshow(generated_image)
        # plt.show()
        return np.uint8(generated_image), np.uint8(generated_label)



    def _get_random_coordinates(self, num=300, margin=6):
        coordinates_x = np.random.choice(self.dims[0] - margin, num)
        coordinates_y = np.random.choice(self.dims[1] - margin, num)
        coordinates = np.vstack((coordinates_x, coordinates_y)).T
        return coordinates





if __name__ == '__main__':
    dg = DataGenerator((1024, 1024))
    dg.sample_image(bound = 0.6, ratio = 0.5,targets=900)
    dg.sample_images(100, targets=300, bound = 0.6, ratio = 0.5, image_dir="images", label_dir="labels")
