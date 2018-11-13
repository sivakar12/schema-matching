import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,\
    precision_score, recall_score
from scipy.optimize import linear_sum_assignment

def make_true_mappings_dataframe(pairs_file_text):
    pairs = [line.strip().split(':') for line in pairs_file_text.split('\n') if line != '']
    rows = [r[0] for r in pairs]
    columns = [r[1] for r in pairs]
    rows, columns
    df = pd.DataFrame(columns=columns, index=rows)
    for pair in pairs:
        df.loc[pair[0], pair[1]] = 1.0
        df.fillna(0.0, inplace=True)
    return df

def make_true_mappings_dict(pairs_file_text):
    pairs = [line.strip().split(':') for line in pairs_file_text.split('\n') if line != '']
    return { pair[0]: pair[1] for pair in pairs }

def get_intersection(true_mappings, pred_mappings):
    rows = set(true_mappings.index).intersection(set(pred_mappings.index))
    columns = set(true_mappings.columns).intersection(set(pred_mappings.columns))
    return true_mappings.loc[rows, columns], pred_mappings.loc[rows, columns]

def apply_hungarian(matrix):
    row_indices, col_indices = linear_sum_assignment(-1 * matrix)
    new_matrix = pd.DataFrame(np.zeros_like(matrix.values), 
                            index=matrix.index, columns=matrix.columns)
    for row, col in zip(row_indices, col_indices):
        new_matrix.iloc[row, col] = 1
    return new_matrix


def mean_difference(true_mappings, pred_mappings):
    true, pred = get_intersection(true_mappings, pred_mappings)
    diff = true.values - pred.values
    return -np.sum(np.abs(diff)) /( diff.shape[0] * diff.shape[1])

# def log_loss(true_mappings, pred_mappings):
#     true_subset, pred_subset = get_intersection(true_mappings, pred_mappings)
#     return -sklearn_log_loss(np.argmax(true_subset.values, axis=1), pred_subset.values)

def accuracy(true_mappings, pred_mappings):
    pred_mappings = apply_hungarian(pred_mappings)
    true_subset, pred_subset = get_intersection(true_mappings, pred_mappings)
    return accuracy_score(true_subset.idxmax(), pred_subset.idxmax())

def precision(true_mappings, pred_mappings):
    pred_mappings = apply_hungarian(pred_mappings)
    true_subset, pred_subset = get_intersection(true_mappings, pred_mappings)
    return precision_score(true_subset.idxmax(), pred_subset.idxmax(), average='micro')

def recall(true_mappings, pred_mappings):
    pred_mappings = apply_hungarian(pred_mappings)
    true_subset, pred_subset = get_intersection(true_mappings, pred_mappings)
    return recall_score(true_subset.idxmax(), pred_subset.idxmax(), average='micro')

def f1(true_mappings, pred_mappings):
    pred_mappings = apply_hungarian(pred_mappings)
    true_subset, pred_subset = get_intersection(true_mappings, pred_mappings)
    return f1_score(true_subset.idxmax(), pred_subset.idxmax(), average='micro')

def print_all_scores(true_mappings, pred_mappings):
    print("Mean difference: ", mean_difference(true_mappings, pred_mappings))
    # print("Log loss: ", log_loss(true_mappings, pred_mappings))
    print("Accuracy: ", accuracy(true_mappings, pred_mappings))
    print("Precision: ", precision(true_mappings, pred_mappings))
    print("Recall: ", recall(true_mappings, pred_mappings))
    print("F1: ", f1(true_mappings, pred_mappings))