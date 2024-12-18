        y_pred.append(similarity)
    y = np.array(y)
    y_pred = np.array(y_pred)
    return y, y_pred

from tqdm import tqdm
from sklearn.metrics import roc_curve



def evaluate(matcher, test_dataloader):
    y = []
    y_pred = []
    for img, label in tqdm(test_dataloader):
        label = label.item()
        _, similarity = matcher.match_train(img)
        y.append(1 if label != -1 else -1)
        y_pred.append(similarity)
    y = np.array(y)
    y_pred = np.array(y_pred)
    return y, y_pred



