import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


class Evaluator:
    """
    Evaluator class
    Functions
    - Evaluates a given model with test input and labels with weighted/unweighted AUC
    - Plot loss curve given a history
    - Plots ROC curve with evaluate function
    """

    def __init__(self, model, binary, save_dir, config_str):
        self.model = model
        self.binary = binary
        self.save_dir = save_dir
        self.config_str = config_str

    def evaluate(self, X_test, y_test, history, show=True, **kwargs):
        # use test dataset for evaluation
        if self.binary:
            y_proba = self.model.predict(X_test)  # outputs two probabilties
            # print(y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
        else:
            # will give error if w_a or w_b doesn't exist
            w_a = kwargs['w_a']
            w_b = kwargs['w_b']
            y_pred_test = self.model.predict(X_test)
            _, w_a_test, _, w_b_test = train_test_split(w_a, w_b, test_size=0.2, random_state=123456)
            auc, y_label_roc, y_pred_roc = self.customROCScore(y_pred_test, w_a_test, w_b_test)
            fpr, tpr, _ = roc_curve(y_label_roc, y_pred_roc, sample_weight=np.r_[w_a_test, w_b_test])
        self.plotROCCurve(fpr, tpr, auc)
        self.plotLoss(history)
        if show:
            plt.show()
        return auc

    def customROCScore(self, pred, w_a, w_b):
        set_a = np.ones(len(pred))
        set_b = np.zeros(len(pred))
        y_pred_roc = np.r_[pred, pred]
        y_label_roc = np.r_[set_a, set_b]
        w_roc = np.r_[w_a, w_b]
        custom_auc = roc_auc_score(y_label_roc, y_pred_roc, sample_weight=w_roc)
        return custom_auc, y_label_roc, y_pred_roc

    def plotLoss(self, history):
        # Extract number of run epochs from the training history
        epochs = range(1, len(history.history["loss"])+1)
        # Extract loss on training and validation ddataset and plot them together
        plt.figure(figsize=(10, 8))
        plt.plot(epochs, history.history["loss"], "o-", label="Training")
        plt.plot(epochs, history.history["val_loss"], "o-", label="Test")
        plt.xlabel("Epochs"), plt.ylabel("Loss")
        # plt.yscale("log")
        plt.legend()
        plt.savefig(f'./{self.save_dir}/fig/loss_{self.config_str}')

    def plotROCCurve(self, fpr, tpr, auc):
        #  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.grid()
        ax.text(0.6, 0.3, 'Custom AUC Score: {:.3f}'.format(auc),
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title('ROC curve')
        plt.savefig(f'{self.save_dir}/fig/ROC_curve_{self.config_str}.PNG')
