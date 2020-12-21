import numpy as np
import json
from scipy import stats
from pandas import read_csv
import matplotlib.pyplot as plt
from reader import read

results = read()

for dataset in ['BBBC004', 'BBBC039']:
    fig, axs = plt.subplots(2, 2)#name + ", " + trans[m])
    fig.suptitle(dataset)
    for m in ['jaccard', 'dice', 'adj_rand', 'warping_error']:
        fcm_vals = results[dataset]['fcm'][m]
        unet_vals = results[dataset]['unet'][m]
        dognet_vals = results[dataset]['dognet'][m]


        trim = 0.0
        trimmed_fcm = stats.trimboth(fcm_vals, trim)
        trimmed_unet = stats.trimboth(unet_vals, trim)
        trimmed_dognet = stats.trimboth(dognet_vals, trim)

        trans = dict(zip(['jaccard', 'dice', 'adj_rand', 'warping_error'], ['Jaccard Index', 'Dice Coefficient', 'Adj. Rand index', 'Warping error']))
        

        if m == 'jaccard':
            p = axs[0][0]
            p.set_xlim([0.7, 1.0])
        elif m == 'dice':
            p = axs[0][1]
            p.set_xlim([0.7, 1.0])
        elif m == 'adj_rand':
            p = axs[1][0]
            p.set_xlim([0.7, 1.0])
        elif m == 'warping_error':
            p = axs[1][1]
            p.set_xlim([0, 0.007])

        
        p.title.set_text(trans[m])
        quantiles = [i for i in np.arange(0, 1.01, 0.1)]
        #print(quantiles)
        b_fcm = np.quantile(trimmed_fcm, quantiles)
        b_unet = np.quantile(trimmed_unet, quantiles)
        b_dognet = np.quantile(trimmed_dognet, quantiles)
        print(b_fcm)

        p.hist(trimmed_fcm, bins=b_fcm, density=True, edgecolor='black', lineStyle='dashed', fc=(1, 0, 0, 0.5), label='csFCM')#, color='red')
        p.hist(trimmed_unet, bins=b_unet, density=True, edgecolor='black', lineStyle='dotted', fc=(0, 0, 1, 0.5), label='U-Net')#, color='red')
        #if 'BBBC039' in fcm:
        p.hist(trimmed_dognet, bins=b_dognet, density=True, edgecolor='black', lineStyle='dashdot', fc=(0, 1, 0, 0.5), label='DoGNet')#, color='red')

        #plt.hist(diff, bins=15, density=True, edgeColor='black', lineStyle='dashed')
        p.set_yticks([])
        p.legend()
    plt.show()
        # Plot histograms