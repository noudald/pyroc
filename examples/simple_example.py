"""Simple example showing how PyROC should be used."""

import pyroc

if __name__ == '__main__':
    ground_truth = [0, 0, 1, 1, 1]
    predictions = [0.1, 0.3, 0.2, 0.3, 0.4]

    roc = pyroc.ROC(ground_truth, predictions)

    print(f'ROC AUC: {roc.auc}')
