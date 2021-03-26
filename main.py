import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/raw.json", help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

print('------------ Options -------------')
for k, v in vars(params).items():
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

np.random.seed(params.seed)
name_dict = {'lr': 'Logistic Regression', 'svm': 'SVM', 'krr': 'Kernel Ridge Regression', 'ksvm': 'Kernel SVM'}
classifer_name = name_dict[params.clf]

# loading data
data = []
labels = []
for k in range(3):
    if params.data_type = 'mat100':
        data[k] = pd.read_csv(Path(params.data_dir, f'Xtr{k}_mat100.csv'))
    else:
        data[k] = pd.read_csv(Path(params.data_dir, f'Xtr{k}.csv'))
    labels[k] = pd.read_csv(Path(params.label_dir, f'Ytr{k}.csv'))

# switch from {0,1} binary labels to {-1,1}
if params.relabel:
    for k in range(3):
        labels[k][labels[k] == 0] = -1

# train/val split
data_train, data_val = [], []
labels_train, labels_val = [], []
for k in range(3):
    (data_train[k], labels_train[k]), (data_val[k], labels_val[k]) = train_val_split(data[k], labels[k], val_size=params.val_size, seed=params.seed):

# train and validate models on each dataset
for k in range(3):
    x_tr, y_tr = data_train[k], labels_train[k]
    x_val, y_val = data_val[k], labels_val[k]
    if params.use_kernel:
        K_tr = get_kernel_matrix(x_tr, x_tr, params)
        K_tr_val = get_kernel_matrix(x_tr, x_val, params)
        lmbda = params.lmbdas[k]
        clf = get_kernel_classsifier(K_tr, params)
        clf.fit(y_tr)
        pred_tr = clf.predict(K_tr)
        pred_val = clf.predict(K_tr_val)
    else:
        clf = get_classsifier(x_tr, y_tr, params)
        clf.fit(x_tr, y_tr)
        pred_tr = clf.predict(x_tr)
        pred_val = clf.predict(x_val)
    acc_tr = accuracy(pred_tr, y_tr)
    acc_val = accuracy(pred_val, y_val)
    print(f'classifier: {classifier_name} | dataset: {k} | training accuracy: {acc_tr} | validation accuracy: {acc_val}')

# get predictions on test datasets
print('generating prediction on test datasets')
test_df = generate_test_predictions_kernel(data, labels, params)
submission_name = get_submission_name(params)
sumbission_path = Path(params.result_dir, submission_name).as_posix()
test_preds_df.to_csv(sumbission_path, index=False)
print(f'result successfully saved at: {sumbission_path}')