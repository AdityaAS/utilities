import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import classification_report

def gt_vs_pred(targets, predictions):
    sorted_ind = np.argsort(targets)
    targets = targets[sorted_ind]
    predictions = predictions[sorted_ind]
    fig = plt.figure()
    plt.plot(targets, targets, c='g')
    plt.scatter(targets, predictions, c='k', s=5)
    plt.xlabel('True Weight')
    plt.ylabel('Predicted Weight')
    plt.legend()

    return fig


def transform_data(data, function):
    return pd.Series(list(map(function, data))).to_numpy()


def compute_classification_report(targets, predictions, target_names):
    class_report = classification_report(targets, predictions, target_names=target_names, output_dict=True)

    del class_report['accuracy']
    del class_report['weighted avg']

    df_classification_report = pd.DataFrame.from_dict(class_report, orient='index')
    df_classification_report['support-pct'] = df_classification_report['support'] / df_classification_report.loc['macro avg', 'support']

    keys = ['precision', 'recall', 'f1-score', 'support-pct']
    functions = [lambda x: round(x, 2), lambda x: round(x, 2), lambda x: round(x, 2), lambda x: round(100 * x, 1)]

    for key, function in zip(keys, functions):
        df_classification_report[key] = transform_data(df_classification_report[key], function)

    data = np.hstack((np.array(df_classification_report.index).reshape((-1, 1)),
        df_classification_report.to_numpy())).tolist()


    columns = [''] + df_classification_report.columns[:].tolist()
    return data, columns

