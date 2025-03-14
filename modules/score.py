import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics
from modules import kaggle_metric_utilities
from sklearn.metrics import f1_score, auc, roc_curve, precision_score, recall_score, average_precision_score

class ParticipantVisibleError(Exception):
    pass

def kaggle_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    
    '''
    Calculates the average of mAP score across each protein-group combination.
    This metric only applies to Leash Bio - Predict New Medicines with BELKA
    https://www.kaggle.com/competitions/leash-BELKA
    Pseudocode:
    1. Calculate mAP at each protein-group combination
    2. Return average of Step 1
    '''

    target = 'binds'

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All target values must be numeric')

    if submission.min().min() < 0:
        raise ParticipantVisibleError('All target values must be at least zero')

    if submission.max().max() > 1:
        raise ParticipantVisibleError('All target values must be no greater than one')

    protein_names = solution['protein_name'].unique().tolist()
    split_groups = solution['split_group'].unique().tolist()

    scores = []

    for protein_name in protein_names:
        for split_group in split_groups:
            select = (solution['protein_name'] == protein_name) & (solution['split_group'] == split_group)
            if len(solution.loc[select]) > 0: # not all combinations are present in the Public split
                score = kaggle_metric_utilities.safe_call_score(
                    sklearn.metrics.average_precision_score,
                    solution.loc[select, target].values,
                    submission.loc[select].values
                )
                print(f'{protein_name} {split_group} {score:.3f}')
                scores.append(score)

    return np.mean(scores)

def print_results(actual, pred_proba, metrics = ['f1', 'gini', 'precision', 'recall', 'average_precision_score']):

    predictions = [1 if x > 0.5 else 0 for x in pred_proba]   

    results = {}

    for metric in metrics:

        if metric == 'f1':
            results['f1'] = round(f1_score(actual, predictions), 3)
        
        elif metric == 'gini':
            fpr, tpr, thresholds = roc_curve(actual, pred_proba, pos_label=1)
            auc_result = auc(fpr, tpr)
            results['gini'] = round(auc_result*2-1, 2)

        elif metric == 'precision':
            results['precision'] = round(precision_score(actual, predictions), 3)

        elif metric == 'recall':
            results['recall'] = round(recall_score(actual, predictions), 3)

        elif metric == 'average_precision_score':
            results['average_precision_score'] = round(average_precision_score(actual, pred_proba), 3)

    print(results)