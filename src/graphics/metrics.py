import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay




def roc_graph(fpr, tpr, y, y_pred):
    RocCurveDisplay(fpr=fpr, tpr=tpr).from_predictions(y, y_pred)
    plt.plot([0, 1], [0, 1], 'k--', label='Identity')
    plt.legend()
    plt.show()
