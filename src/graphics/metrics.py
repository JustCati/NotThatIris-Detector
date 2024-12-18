import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay




def roc_graph(fpr, tpr, y, y_pred):
    RocCurveDisplay(fpr=fpr, tpr=tpr).from_predictions(y, y_pred)
    plt.plot([0, 1], [0, 1], 'k--', label='Identity')
    plt.legend()
    plt.show()


def far_frr_graph(far, frr, threshold, eer_index):
    plt.plot(threshold, far, label="FAR", zorder=-1)
    plt.plot(threshold, frr, label="FRR", zorder=-1)

    eer = far[eer_index]
    plt.axvline(x=threshold[eer_index], color='red', linestyle='--', zorder=0)
    plt.scatter(threshold[eer_index], eer, color="red", label="EER", zorder=1)
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()
