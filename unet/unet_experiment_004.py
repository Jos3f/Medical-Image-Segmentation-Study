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
from datetime import datetime
from multiprocessing.dummy import Pool
import sys

# The H4ck3rs solution when proper organization in modules is too daunting
# N.b. this requires everything to be run from the Unet directory 
sys.path.append("../metrics")
from metrics import Metrics
from threshold_utils import get_best_threshold, normalize_output

# import keras

import tensorflow as tf

# Run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# from keras.backend.tensorflow_backend import set_session
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

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    randVal = tf.random.uniform(())

    if randVal > 0.25:
        k = 1
        if randVal > 0.5:
            k += 1
        if randVal > 0.75:
            k += 1
        input_image = tf.image.rot90(input_image, k=k)
        input_mask = tf.image.rot90(input_mask, k=k)

    return input_image, input_mask


def remove_border(predictions, desired_dim_x, desired_dim_y):
    """
    Remove extra border.
    :param predictions: 4d numpy array of predictions
    :param desired_dim_x:
    :param desired_dim_y:
    :return:
    """
    width = predictions.shape[1]
    height = predictions.shape[2]
    assert desired_dim_x <= width
    assert desired_dim_y <= height
    assert (width - desired_dim_x) % 2 == 0
    assert (height - desired_dim_y) % 2 == 0
    border_x = int((width - desired_dim_x) / 2)
    border_y = int((height - desired_dim_y) / 2)
    predictions = predictions[:, border_x:(width - border_x), border_y:(height - border_y), :]
    return predictions

def main(start_index=0, last_index = 99, filename=None, plot_validation=False, plot_test=True, calculate_train_metric=False):
    """

    :param start_index:
    :param filename:
    :param plot_validation: Plots 3 samples from the validation set each fold
    :param plot_test:  Plots the test test image for each fold
    :return:
    """
    if filename is None:
        now = datetime.now()
        current_dt = now.strftime("%y_%m_%d_%H_%M_%S")
        filename = "results/" + current_dt + ".csv"
    results_file = Path(filename)
    if not results_file.is_file():
        results_file.write_text('index;jaccard;Dice;Adj;Warp;jaccard_to;Dice_to;Adj_to;Warp_to\n')

    """ Load data """
    #image_path = "data/BBBC004_v1_images/*/"
    #label_path = "data/BBBC004_v1_foreground/*/"
    image_path = "../datasets/BBBC004/images/all/"
    label_path = "../datasets/BBBC004/masks/all/"


    file_extension = "tif"

    inp_dim = 950

    file_names = sorted(glob.glob(image_path + "*." + file_extension))
    file_names_labels = sorted(glob.glob(label_path + "*." + file_extension))

    print(file_names)
    print(file_names_labels)

    # Determine largest and smallest pixel values in the dataset
    min_val = float('inf')
    max_val = float('-inf')
    for filename in file_names:
        img = plt.imread(filename)
        if np.min(img) < min_val:
            min_val = np.min(img)
        if np.max(img) > max_val:
            max_val = np.max(img)
        
    images = []
    for file in file_names:
        if file_extension == "tif":
            images.append(tf.convert_to_tensor(np.expand_dims(plt.imread(file), axis=2)))  # For .tif
            #images[-1] = images[-1] / 255  # Normalize
            images[-1] = (images[-1] - min_val) / (max_val - min_val)
            images[-1] = tf.image.resize(images[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')
            #print(np.min(images[-1]), np.max(images[-1]))
        elif file_extension == "png":
            images.append(tf.convert_to_tensor(plt.imread(file)[:, :, :3]))  # For .png
            images[-1] = tf.image.resize(images[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')
            images[-1] = tf.image.rgb_to_grayscale(images[-1])

        images[-1] = mirror_pad_image(images[-1], pixels=21)

    labels = []
    for file in file_names_labels:
        label = plt.imread(file)[:, :, :3]
        label = (np.expand_dims(np.sum(label, axis=2), axis=2))


        label = np.where(label > 0, [0, 1], [1, 0])
        labels.append(tf.convert_to_tensor(label))

        labels[-1] = tf.image.resize(labels[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')
        labels[-1] = np.where(labels[-1] > 0.5, 1, 0)

        labels[-1] = mirror_pad_image(labels[-1], pixels=21)

    print("num images: " + str(len(images)))
    print("num labels: " + str(len(labels)))

    num_data_points = len(images)


    for test_data_point_index in range(start_index, num_data_points):
        if test_data_point_index > last_index:
            break
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

        train_dataset = image_dataset.take(80)
        validation_dataset = image_dataset.skip(80)

        train_dataset.shuffle(80, reshuffle_each_iteration=True)

        train_dataset = train_dataset.map(augment_image) # Apply transformations to training data


        """Load model"""
        print(circles.channels)
        print(circles.classes)

        unet_model = unet.build_model(channels=circles.channels,
                                      num_classes=circles.classes,
                                      layer_depth=3,
                                      filters_root=16)
        if calculate_train_metric:
            unet.finalize_model(unet_model)
        else:
            unet.finalize_model(unet_model,
                            dice_coefficient=False,
                            auc=False,
                            mean_iou=False) # Don't track so many metrics


        """Train"""
        # Use early stopping or not?
        # es_callback = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', 
        #     patience=6, 
        #     restore_best_weights=True)
        trainer = unet.Trainer(checkpoint_callback=False,
                    tensorboard_callback=False,
                    tensorboard_images_callback=False,
                    #callbacks=[es_callback]
        )
        trainer.fit(unet_model,
                    train_dataset,
                    #validation_dataset,
                    epochs=40,
                    batch_size=2
        )

        """Calculate best amplification"""
        prediction = unet_model.predict(validation_dataset.batch(batch_size=1))

        original_images = []
        metric_labels = []
        metric_predictions_unprocessed = []
        metric_predictions = []

        dataset = validation_dataset.map(utils.crop_image_and_label_to_shape((inp_dim, inp_dim, 2)))
        prediction = remove_border(prediction, inp_dim, inp_dim)


        for i, (image, label) in enumerate(dataset):
            original_images.append(image[..., -1])
            metric_labels.append(np.argmax(label, axis=-1))
            metric_predictions_unprocessed.append(normalize_output(prediction[i, ...]))


        best_tau, best_score = get_best_threshold(
            metric_predictions_unprocessed, 
            metric_labels, 
            min=0, max=1, num_steps=50, 
            use_metric=1)

        #best_tau = 0.5 # Use this to not threshold at all, also comment above
        print("Best tau: " + str(best_tau))
        print("Best avg score: " + str(best_score))

        for i in range(len(metric_predictions_unprocessed)):
            metric_predictions.append((metric_predictions_unprocessed[i] >= best_tau).astype(int))

        if plot_validation:
            fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))

            for i in range(3):
                ax[i][0].matshow(original_images[i])
                ax[i][1].matshow(metric_labels[i], cmap=plt.cm.gray)
                ax[i][2].matshow(metric_predictions[i], cmap=plt.cm.gray)

            plt.tight_layout()
            plt.show()

        original_images = []
        metric_labels_test = []
        metric_predictions_unprocessed_test = []
        metric_predictions = []
        metric_predictions_unthresholded = []


        """Evaluate and print to file"""
        prediction = unet_model.predict(test_dataset.batch(batch_size=1))
        dataset = test_dataset.map(utils.crop_image_and_label_to_shape((inp_dim, inp_dim, 2)))
        prediction = remove_border(prediction, inp_dim, inp_dim)
        print("Test shape shape: ", prediction.shape)


        for i, (image, label) in enumerate(dataset):
            original_images.append(image[..., -1])
            metric_labels_test.append(np.argmax(label, axis=-1))
            metric_predictions_unprocessed_test.append(prediction[i, ...])

        for i in range(len(metric_predictions_unprocessed_test)):
            metric_predictions.append((normalize_output(metric_predictions_unprocessed_test[i]) >= best_tau).astype(int))
            metric_predictions_unthresholded.append((normalize_output(metric_predictions_unprocessed_test[i]) >= 0.5).astype(int))

        # Calculate thresholded and unthresholded metrics in parallel
        parallel_metrics = [
            Metrics(
                metric_labels_test, 
                metric_predictions_unthresholded, 
                safe=False,
                parallel=False),

            Metrics(
                metric_labels_test, 
                metric_predictions,
                safe=False,
                parallel=False)
        ]

        def f(m):
            return (
                m.jaccard()[0],
                m.dice()[0],
                m.adj_rand()[0],
                m.warping_error()[0]
            )
        
        pool = Pool(2)
        metric_result = pool.map(f, parallel_metrics)
        
        jaccard_index = metric_result[0][0]
        dice = metric_result[0][1]
        adj = metric_result[0][2]
        warping_error = metric_result[0][3]

        jaccard_index_to = metric_result[1][0]
        dice_to = metric_result[1][1]
        adj_to = metric_result[1][2]
        warping_error_to = metric_result[1][3]

        with results_file.open("a") as f:
            f.write(
                str(test_data_point_index) + ";" + 
                str(jaccard_index) + ";" + 
                str(dice) + ";" + 
                str(adj) + ";" + 
                str(warping_error) + ";" +
                str(jaccard_index_to) + ";" + 
                str(dice_to) + ";" + 
                str(adj_to) + ";" + 
                str(warping_error_to) + "\n"
            )

        print("test_data_point_index: " + str(test_data_point_index))
        print("Jaccard index: " + str(jaccard_index) + " with threshold optimization: " + str(jaccard_index_to))
        print("Dice: " + str(dice) + " with threshold optimization: " + str(dice_to))
        print("Adj: " + str(adj) + " with threshold optimization: " + str(adj_to))
        print("Warping Error: " + str(warping_error) + " with threshold optimization: " + str(warping_error_to))


        """Plot predictions"""
        if plot_test:
            fig, ax = plt.subplots(1, 3, figsize=(8, 4))
            fig.suptitle("Test point: " + str(test_data_point_index), fontsize=14)


            ax[0].matshow(original_images[i])
            ax[0].set_title("Input data")
            ax[0].set_axis_off()

            ax[1].matshow(metric_labels[i], cmap=plt.cm.gray)
            ax[1].set_title("True mask")
            ax[1].set_axis_off()

            ax[2].matshow(metric_predictions[i], cmap=plt.cm.gray)
            ax[2].set_title("Predicted mask")
            ax[2].set_axis_off()


            fig.tight_layout()
            plt.show()


if __name__ == '__main__':
    try:
        results_file = sys.argv[1]
    except IndexError:
        print("No file name given, results file will be given a name automatically")
        results_file = None
    try:
        start_index = int(sys.argv[2])
    except (IndexError, ValueError):
        start_index=0
    try:
        last_index = int(sys.argv[3])
    except (IndexError, ValueError):
        last_index=99
    main(   start_index=start_index, 
            last_index=last_index, 
            filename=results_file, 
            plot_validation=False, 
            plot_test=False, 
            calculate_train_metric=False)
