import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class_names = ['HF presence', 'HF absence']

def eval(model, X, y, output_dir):
    prediction = model.predict(X)

    # plot ROC curve
    plt.figure()
    plot_roc(y, prediction)
    plt.savefig(output_dir + 'roc_curve.png')
    plt.show()


    for i in range(0, len(prediction)):
        if prediction[i][0] > prediction[i][1]: # absence
            prediction[i][0] = 1
            prediction[i][1] = 0
        else: # presence
            prediction[i][0] = 0
            prediction[i][1] = 1
    # Confusion matrix
    cnf_matrix = confusion_matrix(y[:,1], prediction[:,1], labels=[1, 0]) # target = HF presence (value=1)
    # print(cnf_matrix)
    np.set_printoptions(precision=2)

    # indices
    eval_indices(cnf_matrix)

    # plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.savefig(output_dir + 'confusion_matrix.png')
    plt.show()


def eval_indices(cnf_matrix):
    sensitivity = cnf_matrix[1][1] / cnf_matrix[1].sum(axis=0)
    specificity = cnf_matrix[0][0] / cnf_matrix[0].sum(axis=0)
    FP_rate = 1 - specificity
    FN_rate = 1 - sensitivity
    print("Sensitivity: %.2f%%\nSpecificity: %.2f%%" % (sensitivity * 100, specificity * 100))
    print("False positive rate: %.2f%%\nFalse negative rate: %.2f%%" % (FP_rate * 100, FN_rate * 100))
    recall = sensitivity
    precision = cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[0][1])
    f1 = 2 * ((recall * precision) / (recall + precision))
    print("Recall: %.2f%%\nPrecision: %.2f%%\nF1: %.2f%%" % (recall * 100, precision * 100, f1 * 100))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # print(np.sum(cm))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = format(cm[i, j], fmt) +' ('+ format(cm[i, j] / np.sum(cm) * 100, '.2f') + '%)'
        plt.text(j, i, txt,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Estimated label')

def plot_roc(y, prediction):
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve and ROC area for each class
    n_classes=2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:,i], prediction[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        #plt.figure()
        plt.plot(fpr[i], tpr[i], label='%s' % class_names[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
    # plt.show()