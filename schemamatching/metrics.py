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

def get_subset(true_mappings, pred_mappings):
    return pred_mappings.loc[true_mappings.index, true_mappings.columns]

def mean_difference(true_mappings, pred_mappings):
    diff = get_subset(true_mappings, pred_mappings).values - true_mappings.values
    return -np.sum(np.abs(diff)) /( diff.shape[0] * diff.shape[1])

def average_log_loss(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    losses = [log_loss(true_mappings.iloc[i], pred_subset.iloc[i]) for i in range(pred_subset.shape[0])]
    return -1 * sum(losses) / len(losses)

def accuracy(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    return accuracy_score(true_mappings.idxmax(), pred_subset.idxmax())

def precision(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    return precision_score(true_mappings.idxmax(), pred_subset.idxmax(), average='weighted')

def recall(true_mappings, pred_mappings):
    pred_subset = get_subset(true_mappings, pred_mappings)
    return recall_score(true_mappings.idxmax(), pred_subset.idxmax(), average='weighted')

def print_all_scores(true_mappings, pred_mappings):
    print("Mean difference: ", mean_difference(true_mappings, pred_mappings))
    print("Log loss: ", average_log_loss(true_mappings, pred_mappings))
    print("Accuracy: ", accuracy(true_mappings, pred_mappings))
    print("Precision: ", precision(true_mappings, pred_mappings))
    print("Recall: ", recall(true_mappings, pred_mappings))