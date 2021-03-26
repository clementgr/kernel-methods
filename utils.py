import copy
import json
import time
import numpy as np
import collections
from itertools import product
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from classifiers import get_kernel_classsifier

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params

def get_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def train_val_split(x, y, val_size=0.3, seed=21):
    y_train, y_val = train_test_split(y, test_size=val_size, random_state=seed)
    x_train, x_val = x.iloc[y_train.index], x.iloc[y_val.index]
    return (x_train, y_train), (x_val, y_val)


def RBF_kernel(x1, x2, sigma=100):
    K = np.exp(-cdist(x1,x2)/2/sigma/sigma)
    return K


def spectrum_kernel(x1, x2, k=6):
    A_k = {''.join(s): i for i, s in enumerate(product(['A', 'T', 'G', 'C'], repeat=k))}
    phi_1 = np.zeros((len(x1), len(A_k)))
    phi_2 = np.zeros((len(x2), len(A_k)))
    for i, x in enumerate(x1):
        for j in range(len(x) - k + 1):
            phi_1[i][A_k[x[j:(j + k)]]] += 1
    for i, x in enumerate(x2):
        for j in range(len(x) - k + 1):
            phi_2[i][A_k[x[j:(j + k)]]] += 1
    K = phi_1 @ phi_2.T
    K += np.eye(K.shape[0], K.shape[1]) * 1e-10
    return K


def mismatch_kernel(x1, x2, k=6, mismatch=1, mismatch_weight=1):
  
  x1 = x1['seq'].values.tolist()
  x2 = x2['seq'].values.tolist()

  base_letters = ['A', 'T', 'G', 'C']
  A_mismatch = [''.join(s) for s in product(base_letters, repeat=mismatch)]
  assert k > mismatch, 'mismatch dim must be smaller than k-spectrum parameter!'
  A_k = {''.join(s): i for i, s in enumerate(product(base_letters, repeat=k))}
  phi_1 = np.zeros((len(x1), len(A_k)))
  phi_2 = np.zeros((len(x2), len(A_k)))
  
  for i, x in enumerate(x1):
    for j in range(len(x) - k + 1):
      k_gram = x[j:(j + k)]
      k_gram_idx = A_k[k_gram]
      phi_1[i][k_gram_idx] += 1
      new_k_gram = copy.deepcopy(list(k_gram))
      for t in range(len(k_gram)-mismatch+1):
        true_gram = k_gram[t:t+mismatch]
        for mismatched_gram in A_mismatch:
          if mismatched_gram != true_gram:
            new_k_gram[t:t+mismatch] = mismatched_gram
            new_k_gram_string = ''.join(e for e in new_k_gram)
            new_k_gram_idx = A_k[new_k_gram_string]
            phi_1[i][new_k_gram_idx] += mismatch_weight
        new_k_gram = [e for e in k_gram]

  for i, x in enumerate(x2):
    for j in range(len(x) - k + 1):
      k_gram = x[j:(j + k)]
      k_gram_idx = A_k[k_gram]
      phi_2[i][k_gram_idx] += 1
      new_k_gram = [e for e in k_gram]
      for t in range(len(k_gram)-mismatch+1):
        true_gram = k_gram[t:t+mismatch]
        for mismatched_gram in A_mismatch:
          if mismatched_gram != true_gram:
            new_k_gram[t:t+mismatch] = mismatched_gram
            new_k_gram_string = ''.join(e for e in new_k_gram)
            new_k_gram_idx = A_k[new_k_gram_string]
            phi_2[i][new_k_gram_idx] += mismatch_weight
        new_k_gram = [e for e in k_gram]

  K = phi_1 @ phi_2.T
  K += np.eye(K.shape[0], K.shape[1]) * 1e-10
  return K


def get_kernel_matrix(x1, x2, params):
    if params.kernel_type == 'rbf':
        K = RBF_kerne(x1, x2, params.sigma)
    elif params.kernel_type == 'spectrum':
        K = spectrum_kernel(x1, x2, params.k)
    elif params.kernel_type == 'mismatch':
        K = mismatch_kernel(x1, x2, params.k, params.mismatch, params.mismatch_weight)
    else:
        raise KeyError(f'{params.kernel_type} not supported!')
    return K


def generate_test_predictions_kernel(data, labels, params):
    ids = []
    bounds = []
    for i, (x_tr, y_tr) in enumerate(zip(data, labels)):
        
        print(f'processing dataset {i}...')
        start_time = time.time()

        K_tr = get_kernel_matrix(x_tr, x_tr, params)
        clf = get_kernel_classsifier(K_tr, params)
        clf.fit(y_tr)

        if params.data_type = 'mat100':
            test_data_path = Path(params.data_dir, f'Xte{i}_mat100.csv')
        else:
            test_data_path = Path(params.data_dir, f'Xte{i}csv')
        test_df = pd.read_csv(test_data_path, header=None, delimiter=' ')
        ids.extend([f'{1000*i+j}' for j in test_df.index.tolist()])
        
        x_te = test_df.values
        K_te = get_kernel_matrix(x_tr, x_te, params)
        
        preds = clf.predict(K_te)
        bounds.extend([int(p) for p in preds.tolist()])

        end_time = time.time()
        dataset_mins, dataset_secs = get_time(start_time, end_time)
        print(f'done! time taken: {dataset_mins}m {dataset_secs}s)')
    
    test_preds_df = pd.DataFrame({'Id': ids, 'Bound': bounds})
    test_preds_df['Bound'] = test_preds_df['Bound'].apply(lambda b: 0 if b == -1 else b)

    return test_preds_df


def get_submission_name(params):
    if params.kernel_type == 'rbf':
        return f'rbf_sigma={params.sigma}_lambda={params.lmbda}.csv'
    elif params.kernel_type == 'spectrum':
        return f'{params.k}-spectrum_lambda={params.lmbda}.csv'
    elif params.kernel_type == 'mismatch':
        return f'{params.k}-spectrum_{params.mismatch}-mismatch_weight={params.mismatch_weight}_lambda={params.lmbda}.csv'
    else:
        raise KeyError(f'{params.kernel_type} not supported!')