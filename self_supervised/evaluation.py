import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(true_labels, preds_labels):
    confmat = confusion_matrix(true_labels, preds_labels)
    cmatd = ConfusionMatrixDisplay(confmat)
    cmatd.plot()
    plt.grid(b=None)
    plt.show()

def print_classification_report(true_labels, preds_labels):
    print(classification_report(true_labels, preds_labels))