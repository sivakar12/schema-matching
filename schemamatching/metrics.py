import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score

def generate_actual_pairs_dataframe(pairs_file_text):
    pairs = [line.strip().split(':') for line in pairs_file_text.split('\n')]
    rows = [r[0] for r in pairs]
    columns = [r[1] for r in pairs]
    rows, columns
    df = pd.DataFrame(columns=columns, index=rows)
    for pair in pairs:
        df.loc[pair[0], pair[1]] = 1.0
        df.fillna(0.0, inplace=True)
    return df


def mean_difference(results, actual_results):
    columns = actual_results.columns
    rows = actual_results.index
    diff = results.loc[rows, columns].values - actual_results.values
    return -np.sum(np.abs(diff)) /( diff.shape[0] * diff.shape[1])

def average_log_loss(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    losses = [log_loss(true_mappings.iloc[i], pred_subset.iloc[i]) for i in range(pred_subset.shape[0])]
    return sum(losses) / len(losses)

def accuracy(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    return accuracy_score(true_mappings.apply(np.argmax), pred_mappings.apply(np.argmax))

def precision(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    return precision_score(true_mappings.apply(np.argmax), pred_mappings.apply(np.argmax))

def recall(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    return recall_score(true_mappings.apply(np.argmax), pred_mappings.apply(np.argmax))