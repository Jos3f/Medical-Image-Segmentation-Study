import dognet
import numpy as np
import os
import skimage.io
import sys
from sklearn.model_selection import LeaveOneOut
from datetime import datetime
sys.path.append("../metrics")

from metrics import Metrics

sys.path.append("../unet")
from threshold_utils import get_best_threshold, normalize_output

import torch
from pathlib import Path
from multiprocessing.dummy import Pool
import matplotlib.pyplot as plt

def inference(net,image,get_inter = False):
    x = np.expand_dims(image,0)
    vx = torch.from_numpy(x).float().cuda()
    res,inter = net(vx)
    if get_inter:
        return res.data.cpu().numpy(),inter.data.cpu().numpy()
    return res.data.cpu().numpy()

def createDataset(filelist):
    train_images = []
    train_labels = []

    pathImages = "../datasets/BBBC039/images/"
    pathMasks = "../datasets/BBBC039/masks/"
    # Determine the max / min of the dataset
    max_val = float('-inf')
    min_val = float('inf')
    for filename in sorted(filelist):
        img = plt.imread(pathImages + filename[:-3]+"tif")
        if np.max(img) > max_val:
            max_val = np.max(img)
        if np.min(img) < min_val:
            min_val = np.min(img)

    for i, filename in enumerate(sorted(filelist)):
        if i == 110:
            print(filename)

      # load image

        img = plt.imread(pathImages + filename[:-3]+"tif")
        img = (img - min_val) / (max_val - min_val)    
        #load annotation
        orig_masks = skimage.io.imread(pathMasks + filename)
        orig_masks = orig_masks[:,:,0]


        orig_masks[orig_masks > 1] = 1 

        #Append to list
        train_images.append(img)
        train_labels.append(orig_masks)

    return train_images, train_labels

def train_models(train_images, train_labels, validationsize, filename = None):
    if filename is None:
        now = datetime.now()
        current_dt = now.strftime("%y_%m_%d_%H_%M_%S")
        filename = "results/" + current_dt + ".csv"
    results_file = Path(filename)
    if not results_file.is_file():
        results_file.write_text('index;jaccard;Dice;Adj;Warp;jaccard_to;Dice_to;Adj_to;Warp_to\n')



    loo = LeaveOneOut()
    total_splits = loo.get_n_splits(train_images)

    validationsize = validationsize - 1

    for index, (train_index, test_index) in enumerate(loo.split(train_images)):
      if index != 110:
          continue    
      run = True
      print(index)
      while run:
        np.random.shuffle(train_index)

        trainingimages = [train_images[i] for i in train_index[:-validationsize]]
        traininglabels = [train_labels[i] for i in train_index[:-validationsize]]

        validationimages = [train_images[i] for i in train_index[-validationsize:]]
        validationlabels = [train_labels[i] for i in train_index[-validationsize:]]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = dognet.SimpleIsotropic(1,15,5).to(device)
        net.weights_init()

        net, errors = dognet.train_routine(
            net.cuda(),
            dognet.create_generator1C(trainingimages,traininglabels, n=total_splits),
            n_iter=3000,
            margin=5,
            loss='softdice',
            lr=0.0001
        )

        print("The mean loss of last 100 step:",np.mean(errors[-100:]))
        if net != None:

            metric_predictions_unprocessed = []
            for x in validationimages:
                pred = inference(net,x[np.newaxis, :],False)[0][0]
                pred_background = np.ones(pred.shape) - pred
                pred_final = np.stack((pred_background,pred),axis=-1)
                metric_predictions_unprocessed.append(normalize_output(pred_final))


            best_tau, best_score = get_best_threshold(
                metric_predictions_unprocessed,
                validationlabels,
                min=0, max=1, num_steps=50,
                use_metric=1
            )


            print("Best tau: " + str(best_tau))
            print("Best avg score: " + str(best_score))


            #Evaluate on testdata
            test_images = train_images[test_index[0]]
            metric_labels_test = [train_labels[test_index[0]]]

            pred = inference(net,test_images[np.newaxis, :],False)[0][0]


            metric_predictions = [(pred >= best_tau).astype(int)]
            metric_predictions_unthresholded = [(pred >= 0.5).astype(int)]

            plt.imsave("sample110.png", metric_predictions[0], cmap=plt.get_cmap('binary_r'))

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

            pool = Pool(4)
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
                    str(test_index[0]) + ";" +
                    str(jaccard_index) + ";" +
                    str(dice) + ";" +
                    str(adj) + ";" +
                    str(warping_error) + ";" +
                    str(jaccard_index_to) + ";" +
                    str(dice_to) + ";" +
                    str(adj_to) + ";" +
                    str(warping_error_to) + "\n"
                )

            print("test_data_point_index: " + str(test_index[0]))
            print("Jaccard index: " + str(jaccard_index) + " with threshold optimization: " + str(jaccard_index_to))
            print("Dice: " + str(dice) + " with threshold optimization: " + str(dice_to))
            print("Adj: " + str(adj) + " with threshold optimization: " + str(adj_to))
            print("Warping Error: " + str(warping_error) + " with threshold optimization: " + str(warping_error_to))

            run = False




if __name__ == "__main__":
    try:
        results_file = sys.argv[1]
    except IndexError:
        print("No file name given, results file will be given a name automatically")
        results_file = None

    filenames = []

    filenames = os.listdir("../datasets/BBBC039/masks/")

    filenames = sorted(filenames)

    train_images, train_labels = createDataset(filenames)

    train_models(train_images, train_labels, 40, results_file)
