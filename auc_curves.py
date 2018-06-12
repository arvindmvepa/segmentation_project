import numpy as np
import glob
import os
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc as auc_
import matplotlib.pyplot as plt

metric_filename = 'auc_graphs'

f1 = open(metric_filename, 'w')

f1.close()

## load targets and masks once for test set
mask_flat = np.load("mask.npy")
target_flat = np.load("target.npy")

test_auc = []
test_auc_10_fpr = []
test_auc_05_fpr = []
test_auc_025_fpr = []
decision_thresh = .75

it = 770

#cwd = os.getcwd()
suffix = "_"+str(it)+".npy"

## load all the files for an iteration
#search_string = cwd+"/*" + suffix
search_string = "*" + suffix
net_results_files = glob.glob(search_string)
num_networks = len(net_results_files)
net_results_list = []

print(net_results_files)

for result_file in net_results_files:
    prediction_flat = np.load(result_file)
    fpr, tpr, thresholds = roc_curve(target_flat, prediction_flat, sample_weight=mask_flat)
    roc_auc = roc_auc_score(target_flat, prediction_flat, sample_weight=mask_flat)

    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='Curve %s (AUC = %0.5f)' % (result_file, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig("auc.png")