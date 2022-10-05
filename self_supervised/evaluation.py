import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, recall_score, precision_score, precision_recall_fscore_support
)

def plot_confusion_matrix(true_labels, preds_labels):
    confmat = confusion_matrix(true_labels, preds_labels)
    cmatd = ConfusionMatrixDisplay(confmat)
    cmatd.plot()
    plt.grid(b=None)
    plt.show()

def print_classification_report(true_labels, preds_labels):
    print(classification_report(true_labels, preds_labels))

# Since we want the model to perform equally well for both classes despite the imbalance, we use macro-averaged metrics.
# Note that weighted metrics (i.e. weighted average of metrics calculated for each label) is similar to accuracy and hence not a good fit.
def f1Score(true_labels, preds_labels):
    return f1_score(true_labels, preds_labels, average='macro')

def recallScore(true_labels, preds_labels):
    return recall_score(true_labels, preds_labels, average='macro')

def precisionScore(true_labels, preds_labels):
    return precision_score(true_labels, preds_labels, average='macro')

def precisionRecallFscoreSupport(true_labels, preds_labels):
    return precision_recall_fscore_support(true_labels, preds_labels, beta=1.0, average='macro')

def precisionRecallFscoreSupport_each_class_individual(true_labels, preds_labels):
    return precision_recall_fscore_support(true_labels, preds_labels, beta=1.0, average=None)