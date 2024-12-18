import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve



def evaluate(matcher, test_dataloader, train=False):
    y = []
    y_pred = []
    match_fun = matcher.match_train if train else matcher.match

    for img, label in tqdm(test_dataloader):
        label = label.item()
        _, similarity = match_fun(img)
        y.append(1 if label != -1 else -1)
        y_pred.append(similarity)
    y = np.array(y)
    y_pred = np.array(y_pred)
    return y, y_pred


def get_eer(y, y_pred):
    far, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    frr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((frr - far)))
    eer_threshold = threshold[eer_index]
    return far, frr, tpr, threshold, eer_index, eer_threshold
