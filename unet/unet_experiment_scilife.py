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

from metrics import Metrics
from threshold_utils import Threshold


import keras

import tensorflow as tf

# Run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

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

def scilife_data():
    image_path = "data/datafranscilife/input/"
    label_path = "data/datafranscilife/output/"

    file_extension = "tif"

    # inp_dim = 1024

    file_names = sorted(glob.glob(image_path + "*." + file_extension))
    file_names_labels = sorted(glob.glob(label_path + "*." + file_extension))

    images = []
    for file in file_names:
        if file_extension == "tif":
            images.append(tf.convert_to_tensor(np.expand_dims(plt.imread(file), axis=2)))  # For .tif
            images[-1] = images[-1] / np.max(images[-1])  # Normalize
            # images[-1] = tf.image.resize(images[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')

        images[-1] = mirror_pad_image(images[-1], pixels=20)
        
    return images, images

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

def main(filename=None, calculate_train_metric=False):
    """

    :param start_index:
    :param filename:
    :param plot_validation: Plots 3 samples from the validation set each fold
    :param plot_test:  Plots the test test image for each fold
    :return:
    """
    now = datetime.now()
    current_dt = now.strftime("%y_%m_%d_%H_%M_%S")
    if filename is None:
        filename = "results/" + current_dt + ".csv"
    results_file = Path(filename)
    if not results_file.is_file():
        results_file.write_text('index; jaccard; Dice; Adj; Warp\n')

    """ Load data """
    image_path = "data/synthetic/images/"
    label_path = "data/synthetic/labels/"

    file_extension = "tif"

    # inp_dim = 572
    # inp_dim = 200
    # inp_dim = 710
    inp_dim = 1024


    file_names = sorted(glob.glob(image_path + "*." + file_extension))
    file_names_labels = sorted(glob.glob(label_path + "*." + file_extension))

    print(file_names)
    print(file_names_labels)

    images = []
    for file in file_names:
        if file_extension == "tif":
            images.append(tf.convert_to_tensor(np.expand_dims(plt.imread(file), axis=2)))  # For .tif
            images[-1] = images[-1] / 255  # Normalize
            images[-1] = tf.image.resize(images[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')
        elif file_extension == "png":
            images.append(tf.convert_to_tensor(plt.imread(file)[:, :, :3]))  # For .png
            images[-1] = tf.image.resize(images[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')
            images[-1] = tf.image.rgb_to_grayscale(images[-1])

        images[-1] = mirror_pad_image(images[-1], pixels=20)

    labels = []
    for file in file_names_labels:
        label = plt.imread(file)
        # label = plt.imread(file)[:, :, :3]
        label = (np.expand_dims(label, axis=2))


        label = np.where(label > 0, [0, 1], [1, 0])
        labels.append(tf.convert_to_tensor(label))

        labels[-1] = tf.image.resize(labels[-1], [inp_dim, inp_dim], preserve_aspect_ratio=True, method='bilinear')
        labels[-1] = np.where(labels[-1] > 0.5, 1, 0)

        labels[-1] = mirror_pad_image(labels[-1], pixels=20)

    print("num images: " + str(len(images)))
    print("num labels: " + str(len(labels)))

    num_data_points = len(images)

    scilife_images, scilife_labels  = scilife_data()

    # plt.matshow(scilife_images[1][..., -1])
    # plt.show()
    #
    # for i in range(len(scilife_images)):
    #     print(np.max(scilife_images[i]))

    images_temp = images.copy()
    labels_temp = labels.copy()

    """for i in range((5)):
        plt.matshow(images_temp[i][..., -1])
        plt.show()
        plt.matshow(np.argmax(labels_temp[i], axis=-1), cmap=plt.cm.gray)
        plt.show()"""

    print("num images: " + str(len(images_temp)))
    print("num labels: " + str(len(labels_temp)))

    random_permutation = np.random.permutation(len(images_temp))
    images_temp = np.array(images_temp)[random_permutation]
    labels_temp = np.array(labels_temp)[random_permutation]

    image_dataset = tf.data.Dataset.from_tensor_slices((images_temp, labels_temp))

    """Crate data splits"""
    train_dataset = image_dataset.take(100)
    validation_dataset = image_dataset.skip(100)

    train_dataset.shuffle(100, reshuffle_each_iteration=True)

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
        unet.finalize_model(unet_model, dice_coefficient=False, auc=False, mean_iou=False)


    """Train"""
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # trainer = unet.Trainer(checkpoint_callback=False, callbacks=[callback])
    trainer = unet.Trainer(checkpoint_callback=False)

    trainer.fit(unet_model,
                train_dataset,
                epochs=25,
                batch_size=1)

    """Sci Life data prediction"""
    scilife_dataset = tf.data.Dataset.from_tensor_slices((scilife_images, scilife_labels))
    prediction = unet_model.predict(scilife_dataset.batch(batch_size=1))

    original_images = []
    metric_labels = []
    metric_predictions_unprocessed = []
    metric_predictions = []

    dataset = scilife_dataset.map(utils.crop_image_and_label_to_shape((inp_dim, inp_dim, 2)))
    prediction = remove_border(prediction, inp_dim, inp_dim)
    # print("Validation shape after: ", prediction.shape)

    for i, (image, _) in enumerate(dataset):
        original_images.append(image[..., -1])
        metric_predictions_unprocessed.append(prediction[i, ...])

    for i in range(len(metric_predictions_unprocessed)):
        metric_predictions.append(
            np.argmax(metric_predictions_unprocessed[i] * np.array([[[1, 1]]]), axis=-1))

    fig, ax = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(25, 60))

    for i in range(5):
        ax[i][0].matshow(original_images[i])
        ax[i][1].matshow(metric_predictions[i], cmap=plt.cm.gray)
        plt.imsave("results/scilifelab_" + str(current_dt) + "_index_" + str(i) + ".png", metric_predictions[i], cmap=plt.cm.gray)

    plt.tight_layout()
    plt.savefig("results/scilifelab_" + str(current_dt) +".png")
    plt.show()


if __name__ == '__main__':
    main(filename="results/SciLife.csv", calculate_train_metric=False)
