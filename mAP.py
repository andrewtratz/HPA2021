import os
import pandas as pd

import numpy as np
import torch
import random
from HPAutils import *

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(0)
import sys

def mAP(PREDS='', DATASET='custom', XLS=True, return_details=False):
    #ROOT = 'D:\\HPA\\test'

    if XLS:
        truthpaths = {'custom': 'F:\\probabilities_truth_custom.csv',
                      'custom512': 'F:\\todo'}
    else:
        truthpaths = {'custom': 'X:\\truth_submission.csv',
                      'custom512': 'F:\\TestFiles512\\truth_submission.csv'}

    TRUTH = truthpaths[DATASET]

    truth = pd.read_csv(TRUTH).to_numpy()
    if isinstance(PREDS, str):
        preds = pd.read_csv(PREDS).to_numpy()
    else:
        preds = PREDS

    if XLS:
        assert(np.array_equal(truth[:,0:2], preds[:,0:2]))

    stats = []

    if XLS:
        for truth_row, pred_row in zip(truth, preds):
            for i in range(2, len(pred_row)):
                if truth_row[i] == 1.0:
                    stats.append([i-2, 1, pred_row[i]])
                else:
                    stats.append([i-2, 0, pred_row[i]])
    else:
        for truth_row, pred_row in zip(truth, preds):
            bits = truth_row[3].split(' ')
            p_bits = pred_row[3].split(' ')

            assert(len(bits) % 3 == 0)
            assert(len(bits) == len(p_bits))
            for i, bit, p_bit in zip(range(0, len(bits)), bits, p_bits):
                if i % 3 == 0:
                    label = int(bit)
                if i % 3 == 1:
                    value = float(bit)
                    prob = float(p_bit)
                if i % 3 == 2:
                    # Determine value of prediction
                    if value == 1.0: # True
                        stats.append([label, 1, prob])
                    else:
                        stats.append([label, 0, prob])

    # Get all of the confidence data into a dataframe
    stats_df = pd.DataFrame(data=stats, columns=['Label', 'State', 'Confidence'])
    sorted = stats_df.sort_values(by='Confidence', ascending=True)

    aucs = []

    for label in range(0, len(LBL_NAMES)):
        lbl_stats = sorted.loc[sorted['Label'] == label].values

        # True positives starts at the number of total positives and decreases from there
        precision = 0.0
        max_precision = 0.0
        recall = 1.0
        old_recall = 1.0
        old_precision = 0.0
        prior_confidence = 0.0
        auc = 0.0

        unique, indices = np.unique(ar=lbl_stats[:, 2], return_index=True)

        for conf, idx in zip(unique, indices):

            tp = lbl_stats[idx:, 1].sum()
            fn = lbl_stats[:idx, 1].sum()
            fp = (len(lbl_stats) - idx) - tp

            # Calc new precision recall values
            recall = float(tp) / float(tp + fn)
            precision = float(tp) / float(tp + fp)

            if recall < old_recall:
                if precision < max_precision: # Should check for change in recall value in order to update the max
                    precision = max_precision
                else:
                    max_precision = precision

                # Increment AUC
                # Rectangle portion
                rect = old_precision * (old_recall - recall)
                auc += rect

                # Triangle portion
                triangle = 0.5 * (precision - old_precision) * (old_recall - recall)
                auc += triangle

                old_recall = recall
                old_precision = precision

        # Do final (0, 1) point
        rect = old_precision * old_recall
        auc += rect

        # Triangle portion
        triangle = 0.5 * (1 - old_precision) * (old_recall - recall)
        auc += triangle

        if not return_details:
            print('AUC for Label ' + str(label) + ":  " + "{0:.1%}".format(auc))
        aucs.append(auc)

    print("mAP Score: " + "{0:.2%}".format(np.average(np.array(aucs))))
    return aucs


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    mAP()
    print('\nsuccess!')



