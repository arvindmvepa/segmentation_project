import numpy as np
import glob
import os
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc as auc_

metric_filename = 'ensemble_results.txt'

f1 = open(metric_filename, 'w')

f1.close()

## load targets and masks once for test set
mask_flat = np.load("mask.npy")
target_flat = np.load("target.npy")

test_auc = []
test_auc_10_fpr = []
test_auc_05_fpr = []
test_auc_025_fpr = []

## for all iterations
#for it in range(10,2010,10):
for it in range(50,2050,50):
    #cwd = os.getcwd()
    suffix = "_"+str(it)+".npy"
    ## load all the files for an iteration
    #search_string = cwd+"/*" + suffix
    search_string = "*" + suffix
    net_results_files = glob.glob(search_string)
    num_networks = len(net_results_files)
    net_results_list = []

    print(net_results_files)
    if len(net_results_files) == 0:
        continue

    for result_file in net_results_files:
        prediction_flat = np.load(result_file)
        net_results_list += [prediction_flat]

    ## process accumulation of results
    ## mean, median, other methods
    ## Code missing, to include

    #prediction_flat = np.array(net_results_list).mean(0)
    prediction_flat = np.array(net_results_list).median(0)

    ## produce results per iteration

    auc = roc_auc_score(target_flat, prediction_flat, sample_weight=mask_flat)
    fprs, tprs, thresholds = roc_curve(target_flat, prediction_flat, sample_weight=mask_flat)
    np_fprs, np_tprs, np_thresholds = np.array(fprs).flatten(), np.array(tprs).flatten(), np.array(thresholds).flatten()
    fpr_10 = np_fprs[np.where(np_fprs < .10)]
    tpr_10 = np_tprs[0:len(fpr_10)]

    fpr_05 = np_fprs[np.where(np_fprs < .05)]
    tpr_05 = np_tprs[0:len(fpr_05)]

    fpr_025 = np_fprs[np.where(np_fprs < .025)]
    tpr_025 = np_tprs[0:len(fpr_025)]

    auc_10_fpr = auc_(fpr_10, tpr_10)
    auc_05_fpr = auc_(fpr_05, tpr_05)
    auc_025_fpr = auc_(fpr_025, tpr_025)


    prediction_flat = np.round(prediction_flat)
    target_flat = np.round(target_flat)

    (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                          prediction_flat,
                                                                          average='binary',
                                                                          sample_weight=mask_flat)

    kappa = cohen_kappa_score(target_flat, prediction_flat, sample_weight=mask_flat)
    tn, fp, fn, tp = confusion_matrix(target_flat, prediction_flat, sample_weight=mask_flat).ravel()

    specificity = tn / (tn + fp)

    f1 = open(metric_filename, 'a')

    test_auc.append((auc, it))
    test_auc_10_fpr.append((auc_10_fpr, it))
    test_auc_05_fpr.append((auc_05_fpr, it))
    test_auc_025_fpr.append((auc_025_fpr, it))
    max_auc = max(test_auc)
    max_auc_10_fpr = max(test_auc_10_fpr)
    max_auc_05_fpr = max(test_auc_05_fpr)
    max_auc_025_fpr = max(test_auc_025_fpr)

    f1.write(
        'Step {}, recall {}, specificity {}, auc {}, auc_10_fpr {}, auc_05_fpr {}, auc_025_fpr {}, precision {}, fbeta_score {}, kappa {}, max auc {} {}, max auc 10 fpr {} {}, max auc 5 fpr {} {}, max auc 2.5 fpr {} {} \n'.format(
            it, recall, specificity, auc, auc_10_fpr, auc_05_fpr, auc_025_fpr, precision, fbeta_score, kappa, max_auc[0], max_auc[1], max_auc_10_fpr[0], max_auc_10_fpr[1], max_auc_05_fpr[0], max_auc_05_fpr[1], max_auc_025_fpr[0], max_auc_025_fpr[1]))
    f1.close()
