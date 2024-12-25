import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay




def roc_graph(fpr, tpr, y, y_pred):
    RocCurveDisplay(fpr=fpr, tpr=tpr).from_predictions(y, y_pred)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], 'k--', label='Identity')
    plt.legend()
    plt.show()



def far_frr_graph(far, frr, threshold, eer_index, eval_far, eval_frr):
    plt.plot(threshold, far, label="FAR", zorder=-1)
    plt.plot(threshold, frr, label="FRR", zorder=-1)

    if eer_index is not None:
        eer = far[eer_index]
        plt.axvline(x=threshold[eer_index], color='red', linestyle='--', zorder=0)
        plt.scatter(threshold[eer_index], eer, color="red", label="EER", zorder=1)

        string = f"FAR at EER: {far[eer_index]:.2f}\n FRR at EER: {frr[eer_index]:.2f}\n Threshold: {threshold[eer_index]:.2f}"
        plt.gca().add_artist(plt.text(0.5, 0.95, string, transform=plt.gca().transAxes,
                          fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5)))
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.title("False Acceptance Rate (FAR) and False Rejection Rate (FRR)")
    string = f"Evaluation: FAR =  {eval_far[eer_index]:.4f}, FRR = {eval_frr[eer_index]:.4f}, Ideal threshold at EER = {eer:.4f}"
    plt.figtext(0.5, 0.01, string, ha="center", va="bottom", fontsize=10)
    plt.show()
