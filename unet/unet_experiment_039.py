import matplotlib.pyplot as plt
import numpy as np
import unet
from unet import utils
from unet.datasets import circles
import os
import glob
import matplotlib.pyplot as plt
import random
from pathlib import Path
import re


from metrics import Metrics
from threshold_utils import Threshold

import keras

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)

"""Helper functions"""

def mirror_pad_image(image, pixels=20):
    image = np.lib.pad(image, ((pixels, pixels), (pixels, pixels), (0,0)), 'reflect')
    return image


@tf.function
def augment_image(input_image, input_mask):
    # input_image = datapoint[0]
    # input_mask = datapoint[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    # randVal = tf.random.uniform(())
    # if randVal > 0.25:
    #     k = 1
    #     if randVal > 0.5:
    #         k += 1
    #     if randVal > 0.75:
    #         k += 1
    #     input_image = tf.image.rot90(input_image, k=k)
    #     input_mask = tf.image.rot90(input_mask, k=k)

    return input_image, input_mask

def split_into_classes(image):

    def one_hot(list):
        if np.sum(list) == 0:
            return [1, 0]
        return [0, 1]

    # matrix = np.where(image != 0, 1, 0)
    # matrix = np.apply_along_axis(one_hot, axis=2, arr=matrix)

    matrix = (np.expand_dims(np.sum(image, axis=2), axis=2))
    matrix = np.where(matrix > 0, [0, 1], [1, 0])



    return matrix

def binarize(image):
    # Turn into binary array & remove alpha channel
    return (np.linalg.norm(image[:, :, :3], axis=(2)) != 0).astype(int)

def read_data():
    image_path = "data/BBBC039/images/"
    label_path = "data/BBBC039/masks/"


    image_dir = (image_path)
    mask_dir = (label_path)

    # Get the images and their corresponding masks
    # image_paths = glob.glob(os.path.join(image_dir, '*'))
    # mask_paths = glob.glob(os.path.join(mask_dir, '*'))
    # Check that all masks have the same file ending
    # mask_pattern = re.compile(".*?(\.(tif|tiff|jpg|jpeg|png))")
    # endings = {}
    # for mp in mask_paths:
    #     ending = mask_pattern.match(mp).group(1)
    #     endings[ending] = True
    # assert (len(endings.keys()) == 1)
    # mask_ending = list(endings.keys())[0]
    #
    # image_ids = []
    # mask_names = []
    #
    # # Extract mask names
    # image_pattern = re.compile("(.*?)\.(tif|tiff|jpg|jpeg|png)")
    # for i in image_paths:
    #     image_name = image_pattern.match(i).group(1)
    #     image_ids.append(image_name)
    #     mask_names.append(os.path.join(mask_dir, (image_name + mask_ending)))

    image_paths = sorted(glob.glob(image_path + "*." + "tif"))
    mask_names = sorted(glob.glob(label_path + "*." + "png"))

    images = [plt.imread(image) for image in image_paths]
    masks = [plt.imread(mask) for mask in mask_names]
    # masks = [binarize(mask) for mask in masks]

    return images, masks

def main(start_index=0, plot=True):
    results_file = Path("results/BBBC039_LOOCV.csv")
    if not results_file.is_file():
        results_file.write_text('index; jaccard; Dice; Adj; Warp\n')

    """ Load data """
    # image_path = "data/BBBC004_v1_images/synthetic_000_images/"
    # label_path = "data/BBBC004_v1_foreground/synthetic_000_foreground/"
    # image_path = "data/BBBC004_v1_images/*/"
    # label_path = "data/BBBC004_v1_foreground/*/"
    #
    # file_extension = "tif"
    #
    # # inp_dim = 572
    # # inp_dim = 200
    # inp_dim = 500
    #
    #
    # file_names = sorted(glob.glob(image_path + "*." + file_extension))
    # file_names_labels = sorted(glob.glob(label_path + "*." + file_extension))
    #
    # print(file_names)
    # print(file_names_labels)
    print("Start read")

    images, labels = read_data()
    print("Done read")

    images = [np.expand_dims(image, axis=2)/ max(np.max(image), 255) for image in images]
    labels = [split_into_classes(label[:,:,:2]) for label in labels]
    # labels = [(label[:,:,:2]) for label in labels]

    print(np.array(images).shape)
    print(np.array(labels).shape)

    # print(np.max(images[5]))

    # for i in range((3)):
    #     plt.matshow(images[i][..., -1])
    #     plt.show()
    #     plt.matshow(np.argmax(labels[i], axis=-1), cmap=plt.cm.gray)
    #     plt.show()
    #
    # for i in images:
    #     print(np.max(i))
    # print(labels[0])
    # print(np.sum(labels[0]) / np.max(labels[0]))
    # non_zero = np.argwhere(labels[0] > 0)
    # print(non_zero)
    # print(np.count_nonzero(labels[0]))
    #
    # for row in labels[0]:
    #     for b in row:
    #         if np.sum(b) > 0:
    #             print(b)



    # assert False

    for i in range(len(images)):
        images[i] = mirror_pad_image(images[i])
        labels[i] = mirror_pad_image(labels[i])


    print("num images: " + str(len(images)))
    print("num labels: " + str(len(labels)))

    num_data_points = len(images)


    for test_data_point_index in range(start_index, num_data_points):
        print("\nStarted for data_point_index: " + str(test_data_point_index))

        images_temp = images.copy()
        labels_temp = labels.copy()

        """for i in range((5)):
            plt.matshow(images_temp[i][..., -1])
            plt.show()
            plt.matshow(np.argmax(labels_temp[i], axis=-1), cmap=plt.cm.gray)
            plt.show()"""

        test_image = images_temp.pop(test_data_point_index)
        test_label = labels_temp.pop(test_data_point_index)

        test_dataset = tf.data.Dataset.from_tensor_slices(([test_image],
                                                              [test_label]))

        print("num images: " + str(len(images_temp)))
        print("num labels: " + str(len(labels_temp)))

        random_permutation = np.random.permutation(len(images_temp))
        images_temp = np.array(images_temp)[random_permutation]
        labels_temp = np.array(labels_temp)[random_permutation]

        image_dataset = tf.data.Dataset.from_tensor_slices((images_temp, labels_temp))

        """Crate data splits"""
        data_augmentation = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        # image_dataset.shuffle(100, reshuffle_each_iteration=False)

        train_dataset = image_dataset.take(160)
        validation_dataset = image_dataset.skip(160)

        train_dataset.shuffle(160, reshuffle_each_iteration=True)

        train_dataset = train_dataset.map(augment_image) # Apply transformations to training data


        """Load model"""
        print(circles.channels)
        print(circles.classes)

        unet_model = unet.build_model(channels=circles.channels,
                                      num_classes=circles.classes,
                                      layer_depth=3,
                                      filters_root=16)
        unet.finalize_model(unet_model)


        """Train"""
        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        # trainer = unet.Trainer(checkpoint_callback=False, callbacks=[callback])
        trainer = unet.Trainer(checkpoint_callback=False)
        trainer.fit(unet_model,
                    train_dataset,
                    validation_dataset,
                    epochs=25,
                    batch_size=1)


        """Calculate best amplification"""
        prediction = unet_model.predict(validation_dataset.batch(batch_size=1))

        original_images = []
        metric_labels = []
        metric_predictions_unprocessed = []
        metric_predictions = []

        dataset = validation_dataset.map(utils.crop_image_and_label_to_shape(prediction.shape[1:]))

        for i, (image, label) in enumerate(dataset):
            original_images.append(image[..., -1])
            metric_labels.append(np.argmax(label, axis=-1))
            metric_predictions_unprocessed.append(prediction[i, ...])


        threshold_util = Threshold(metric_labels)
        best_tau, best_score = threshold_util.get_best_threshold(metric_predictions_unprocessed, min=0, max=2, num_steps=200, metric=0)
        print("Best tau: " + str(best_tau))
        print("Best avg score: " + str(best_score))

        for i in range(len(metric_predictions_unprocessed)):
            metric_predictions.append(np.argmax(metric_predictions_unprocessed[i] * np.array([[[1, best_tau]]]), axis=-1))

        if plot:
            fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))

            for i in range(3):
                ax[i][0].matshow(original_images[i])
                ax[i][1].matshow(metric_labels[i], cmap=plt.cm.gray)
                ax[i][2].matshow(metric_predictions[i], cmap=plt.cm.gray)

            plt.tight_layout()
            plt.show()

        original_images = []
        metric_labels = []
        metric_predictions_unprocessed_test = []
        metric_predictions = []


        """Evaluate and print to file"""
        prediction = unet_model.predict(test_dataset.batch(batch_size=1))
        dataset = test_dataset.map(utils.crop_image_and_label_to_shape(prediction.shape[1:]))

        for i, (image, label) in enumerate(dataset):
            original_images.append(image[..., -1])
            metric_labels.append(np.argmax(label, axis=-1))
            metric_predictions_unprocessed_test.append(prediction[i, ...])


        for i in range(len(metric_predictions_unprocessed_test)):
            metric_predictions.append(np.argmax(metric_predictions_unprocessed_test[i] * np.array([[[1, best_tau]]]), axis=-1))

        quantitative_metrics = Metrics(metric_labels, metric_predictions)

        jaccard_index = quantitative_metrics.jaccard()
        dice = quantitative_metrics.dice()
        adj = quantitative_metrics.adj_rand()
        warping_error = quantitative_metrics.warping_error()
        #warping_error = [0.1]

        with results_file.open("a") as f:
            f.write(str(test_data_point_index) + ";" + str(jaccard_index[0]) + ";" + str(dice[0])
                    + ";" + str(adj[0]) + ";" + str(warping_error[0]) + "\n")

        print("test_data_point_index: " + str(test_data_point_index))
        print("Jaccard index: " + str(jaccard_index))
        print("Dice: " + str(dice))
        print("Adj: " + str(adj))
        print("Warping Error: " + str(warping_error))


        """Plot predictions"""
        if plot:
            fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))

            for i in range(len(metric_labels)):
                ax[i][0].matshow(original_images[i])
                ax[i][1].matshow(metric_labels[i], cmap=plt.cm.gray)
                ax[i][2].matshow(metric_predictions[i], cmap=plt.cm.gray)

            plt.tight_layout()
            plt.show()

        np.save("results/BBBC039_val_fold_" + str(test_data_point_index) + ".npy", metric_predictions_unprocessed)
        np.save("results/BBBC039_test_fold_" + str(test_data_point_index) + ".npy", metric_predictions_unprocessed_test)


if __name__ == '__main__':
    main(start_index=0, plot=False)
