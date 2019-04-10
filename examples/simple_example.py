"""Simple example showing how PyROC should be used."""

import pyroc

if __name__ == '__main__':
    GROUND_TRUTH = [0, 0, 1, 1, 1]
    PREDICTIONS = [0.1, 0.3, 0.2, 0.3, 0.4]

    ROC = pyroc.ROC(GROUND_TRUTH, PREDICTIONS)

    print(f'ROC AUC: {ROC.auc}')
    ROC.plot()
