from reader import read
import numpy as np
from scipy.stats import median_absolute_deviation, kruskal, kstest, ks_2samp, f_oneway, mannwhitneyu, norm, wilcoxon, shapiro, spearmanr

results = read()

open('correlation.txt', 'w').close()
open('results.txt', 'w').close()
open('normality.txt', 'w').close()
open('tests.txt', 'w').close()



def results_formater(vals, name):
    return (
        name + " & " + 
        ' & '.join(list(map(
            lambda x: (("\\num{" + "{:0.2e}".format(np.median(vals[x])) + "}") if x == 'warping_error' else (str(round(np.median(vals[x]), 3)))) + " $\pm$ \\num{" + "{:0.1e}".format(median_absolute_deviation(vals[x])) + "}",
            ['jaccard', 'dice', 'adj_rand', 'warping_error']))).replace("e", "d") + 
        '\\\\ \n'
    )


for dataset in ['BBBC004', 'BBBC039']:
    print("DATASET:", dataset)
    l = []
    
    for method in ['fcm', 'unet', 'dognet']:
        print("Spearman R", method)
        a = np.array(
            [
                results[dataset][method]['jaccard'], 
                results[dataset][method]['dice'], 
                results[dataset][method]['adj_rand'],
                results[dataset][method]['warping_error']
            ]
        ).transpose()
        l.append(a)
        results_matrix = spearmanr(a)[0]
        print(results_matrix)
        with open('correlation.txt', 'a') as file: 
            file.write(dataset + " " + method + "\n")
            for row in results_matrix:
                if dataset == 'BBBC004':
                    file.write("\hline\n")
                file.write(" & ".join(list(map(lambda x: str(round(x, 2)), row))) + " \\\\ \n")
                if dataset == 'BBBC039':
                    file.write("\hline\n")        
            file.write("\n\n\n")
    
    
    with open('results.txt', 'a') as file:
            file.write(dataset + '\n')
            # FCM 
            fcm_vals = results[dataset]['fcm']
            file.write(results_formater(fcm_vals, "csFCM"))

            # U-Net
            unet_vals = results[dataset]['unet']
            file.write(results_formater(unet_vals, "U-Net"))

            # DoGNet
            dognet_vals = results[dataset]['dognet']
            file.write(results_formater(dognet_vals, "DoGNet"))
    
    
    for m in ['jaccard', 'dice', 'adj_rand', 'warping_error']:
        fcm_vals = results[dataset]['fcm'][m]
        unet_vals = results[dataset]['unet'][m]
        dognet_vals = results[dataset]['dognet'][m]

        print(dataset, m)
        print("\tfcm")
        print("\t\tMean:", round(np.median(fcm_vals), 6), "std:", round(np.std(fcm_vals), 6))
        print("\tunet")
        print("\t\tMean:", round(np.median(unet_vals), 6), "std:", round(np.std(unet_vals), 6))
        print("\tdognet")
        print("\t\tMean:", round(np.median(dognet_vals), 6), "std:", round(np.std(dognet_vals), 6))


        with open('normality.txt', 'a') as file:
            vals = [fcm_vals, unet_vals, dognet_vals]
            file.write(dataset + " " + m + '\n')
            for i, val in enumerate(vals):
                file.write(
                    ((["csFCM", "U-Net", "DoGNet"][i] + " & ") if m == "jaccard" else ("")) + 
                    " & ".join(
                        list(map(
                            lambda j: (("\\num{" + "{:0.1e}".format(shapiro(vals[i] - vals[j])[1]) + "}") if i != j else ("-")), 
                            [0, 1, 2]
                            )
                        )
                    ) + " \\\\ \n"
                )
            file.write("\n\n\n")


        print("p-value for Shapiro-Wilk test of normality:")
        for i, vals in enumerate(['fcm', 'unet', 'dognet']):
            for j, vals2 in enumerate(['fcm', 'unet', 'dognet']):
                if i == j:
                    continue
                
                if i == 0:
                    first = fcm_vals
                elif i == 1:
                    first = unet_vals
                else:
                    first = dognet_vals
                
                if j == 0:
                    second = fcm_vals 
                elif j == 1:
                    second = unet_vals
                else:
                    second = dognet_vals
                print("\t", vals, "vs", vals2)
                print("\t", shapiro(first - second)[1])

        print("Wilcoxon test")
        if m != 'warping_error':
            #print("\tUNet - FCM (null H: No difference):", wilcoxon(unet_vals, fcm_vals))
            print("\tAlt. H: U-Net better than FCM):", wilcoxon(unet_vals, fcm_vals, alternative='greater'))
            print("\tAlt. H: FCM better than DoGNet):", wilcoxon(fcm_vals, dognet_vals, alternative='greater'))

        else:
            #print("\tUNet - FCM (null H: No difference):", wilcoxon(unet_vals, fcm_vals))
            print("\tAlt. H: Unet better than FCM):", wilcoxon(unet_vals, fcm_vals, alternative='less'))
            print("\tAlt. H: FCM better than DoGNet):", wilcoxon(fcm_vals, dognet_vals, alternative='less'))

    with open('tests.txt', 'a') as file:
        file.write(dataset + "\n")
        if dataset == 'BBBC004':
            # Four hypotheses
            # U-Net better than DoGNet in terms of three metrics
            file.write(
                "U-Net > DoGNet & " +
                " & ".join(
                    list(
                        map(
                            lambda x: (("-") if x == 'warping_error' else ("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['unet'][x], results[dataset]['dognet'][x], alternative='greater')[1]) + "}")),
                            ['jaccard', 'dice', 'adj_rand', 'warping_error']
                        )
                    )
                ) + 
                " \\\\ \n"
            ) 
            # U-Net better than csFCM in terms of warping_error
            file.write(
                "U-Net > csFCM & " +
                " & ".join(
                    list(
                        map(
                            lambda x: (("-") if x != 'warping_error' else ("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['unet'][x], results[dataset]['fcm'][x], alternative='less')[1]) + "}")),
                            ['jaccard', 'dice', 'adj_rand', 'warping_error']
                        )
                    )
                ) + 
                " \\\\ \n"
            ) 
            # DoGNet performs better than csFCM in terms of three metrics
            file.write(
                "DoGNet > csFCM & " +
                " & ".join(
                    list(
                        map(
                            lambda x: (("-") if x == 'warping_error' else("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['dognet'][x], results[dataset]['fcm'][x], alternative='greater')[1]) + "}")),
                            ['jaccard', 'dice', 'adj_rand', 'warping_error']
                        )
                    )
                ) + 
                " \\\\ \n"
            )
            # csFCM performs better than DoGNet in terms of Warping error
            file.write(
                "csFCM > DoGNet & " +
                " & ".join(
                    list(
                        map(
                            lambda x: (("-") if x != 'warping_error' else("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['fcm'][x], results[dataset]['dognet'][x], alternative='less')[1]) + "}")),
                            ['jaccard', 'dice', 'adj_rand', 'warping_error']
                        )
                    )
                ) + 
                " \\\\ \n"
            )
            file.write("\n\n\n")
        else:
            file.write(dataset + "\n")
            # U-Net better than csFCM in terms of all metrics
            file.write(
                "U-Net > csFCM & " +
                " & ".join(
                    list(
                        map(
                            lambda x: (("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['unet'][x], results[dataset]['fcm'][x], alternative='less')[1]) + "}") if x == 'warping_error' else ("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['unet'][x], results[dataset]['fcm'][x], alternative='greater')[1]) + "}")),
                            ['jaccard', 'dice', 'adj_rand', 'warping_error']
                        )
                    )
                ) + 
                " \\\\ \n"
            ) 

            # csFCM better than DoGNet in terms of all metrics
            file.write(
                "csFCM > DoGNet & " +
                " & ".join(
                    list(
                        map(
                            lambda x: (("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['fcm'][x], results[dataset]['dognet'][x], alternative='less')[1]) + "}") if x == 'warping_error' else ("\\num{" + "{:0.1e}".format(wilcoxon(results[dataset]['fcm'][x], results[dataset]['dognet'][x], alternative='greater')[1]) + "}")),
                            ['jaccard', 'dice', 'adj_rand', 'warping_error']
                        )
                    )
                ) + 
                " \\\\ \n"
            ) 