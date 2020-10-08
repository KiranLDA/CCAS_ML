import numpy as np
import pickle
import os
from itertools import chain
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from evaluation.fragments import get_call_ranges, get_fragments, plot_fragments

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

model = load_model('../saved_models/model_2019-11-25_11:54:19.145467_network_train/savedmodel.h5')
x_test = np.load('/cache/rmishra/dataset/x_test.npy')
y_test = np.load('/cache/rmishra/dataset/y_test.npy')
test_files = np.load('/cache/rmishra/dataset/test_files.npy')


def IoU(predicted, actual):
    overlap = np.sum(np.logical_and(predicted == 1, actual == 1))
    union = np.sum(np.logical_or(predicted == 1, actual == 1))
    IoU = overlap / union
    return IoU

def plot_confusion_matrix(y_test, preds_w_threshold):
    class_names = ['GIG', 'SQL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP', 'OTH']
    preds_w_threshold[preds_w_threshold > 0] = 1
    y_test_ravel = np.argmax(y_test, axis=2).ravel()
    y_pred_ravel = np.argmax(preds_w_threshold, axis=2).ravel()
    cf = confusion_matrix(y_test_ravel, y_pred_ravel)
    df_cm = DataFrame(cf, index=class_names, columns=class_names)
    ax = sns.heatmap(df_cm, cmap="Oranges", annot=True, fmt='g')
    ax.set_title("Confusion matrix for label predictions")
    plt.savefig('Confusion_matrix.png')
    plt.show()
    plt.close()


preds_w_threshold = model.predict(x_test, batch_size=16)

# Considering noise label as the threshold
for idx, frame in enumerate(preds_w_threshold):
    for row in frame:
        row[:8][row[:8] < row[8]] = 0  # making every sound with less than noise amplitude equal to 0

class_names = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
indices = get_call_ranges(y_test)
pred_indices = get_call_ranges(preds_w_threshold)

# plot_confusion_matrix(y_test, preds_w_threshold)
# Performing scoring metrics only on the call labels, and not on the Noise Data.
preds_w_threshold = preds_w_threshold[:, :, :8]
actual = y_test[:, :, :8]

silence_details = get_fragments(y_test, preds_w_threshold, indices)[0]
fragment_details = get_fragments(y_test, preds_w_threshold, indices)[1]


for label_idx in range(fragment_details.shape[1]):
    plot_fragments(fragment_details[:,label_idx], class_names[label_idx])


IoUrecord = []
for i in range(len(x_test)):
    IoUrecord.append(IoU(preds_w_threshold[i], actual[i]))
MeanIoU = sum(IoUrecord) / len(IoUrecord)
print(MeanIoU)

preds_w_threshold[preds_w_threshold > 0] = 1
TP = np.sum(np.logical_and(preds_w_threshold == 1, actual == 1))
FN = np.sum(np.logical_and(preds_w_threshold == 0, actual == 1))
TN = np.sum(np.logical_and(preds_w_threshold == 0, actual == 0))
FP = np.sum(np.logical_and(preds_w_threshold == 1, actual == 0))
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
print("True Positive Rate: " + str(np.round(TPR, 3)))
print("True Negative Rate: " + str(np.round(TNR, 3)))
print("False Positive Rate: " + str(np.round(FPR, 3)))
print("False Negative Rate: " + str(np.round(FNR, 3)))

predicted = model.predict(x_test)
actual = y_test

fpr, tpr, thresholds = roc_curve(actual.ravel(), predicted.ravel(), pos_label=None)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic Overall')
plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ROC_overall' + '.png')
plt.show()
plt.close()


for i in range(len(class_names)):
    fpr, tpr, thresholds = roc_curve(actual[:, :, i].ravel(), predicted[:, :, i].ravel(), pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic ' + class_names[i])
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('class_ROC_' + class_names[i] + '.png')
    plt.show()
    plt.close()
