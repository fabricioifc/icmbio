import sys
sys.path.append('D:\\Projetos\\icmbio\\')
import matplotlib
matplotlib._log.disabled = True

import pandas as pd
import logging, os, io
from pathlib import Path
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from trainer import Trainer
from utils import convert_to_color
import torch
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy
from sklearn import metrics

RESULTS_PATH = './results'

logging.basicConfig(filename=f'{RESULTS_PATH}/log.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Turn off sina logging
for name in ["matplotlib", "matplotlib.font", "matplotlib.pyplot"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True

class TestTrained:

    def __init__(self, test_result_path: str, classes):
        self.test = np.load(test_result_path, allow_pickle=True)
        self.classes = classes
        Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        
        logging.info('\n\n--- Iniciando o teste para gerar relatórios ---')
        logging.info(f'--- {test_result_path} ---')

        print(f'--- Carregando o arquivo com as previsões e ground truths ---')
        self.y_pred = self.test.f.arr_0.item().get('all_preds')
        self.y_true = self.test.f.arr_0.item().get('all_gts')
        
        self.acc = self.test.f.arr_0.item().get('acc')
        logging.info(f'--- Accuracy: {self.acc} ---')
    
    def run(self):
        logging.info(f'--- Geranco a Matriz de Confusão ---')
        cm = confusion_matrix(
            np.concatenate([p.ravel() for p in self.y_true]),
            np.concatenate([p.ravel() for p in self.y_pred]),
            labels=range(len(self.classes)),
        )
        
        fig, ax = plot_confusion_matrix(
            conf_mat=cm, 
            figsize=(8,8), 
            cmap=plt.cm.Greens,
            colorbar=True,
            show_absolute=False,
            show_normed=True,
        )

        if self.classes is not None:
            tick_marks = np.arange(len(self.classes))
            plt.xticks(tick_marks, self.classes, rotation=45)
            plt.yticks(tick_marks, self.classes)

        plt.xlabel('Predictions', fontsize=12)
        plt.ylabel('Target', fontsize=12)
        plt.title('Confusion Matrix', fontsize=12)
        # plt.show()
        logging.info(f'--- Exportando Matriz de Confusão ---')
        fig.savefig(f"{RESULTS_PATH}/cm_all", dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

        logging.info(f'--- Convertendo Array para Tensor ---')
        y_true_ = torch.from_numpy(np.concatenate([p.ravel() for p in self.y_true]))
        y_pred_ = torch.from_numpy(np.concatenate([p.ravel() for p in self.y_pred]))

        accuracy = Accuracy(task="multiclass", num_classes=len(self.classes))
        # DOC: IOU = true_positive / (true_positive + false_positive + false_negative)
        jaccard = MulticlassJaccardIndex(task="multiclass", num_classes=len(self.classes), average='macro')
        acc = accuracy(y_true_, y_pred_)
        jac = jaccard(y_true_, y_pred_)
        jac_all = metrics.jaccard_score(y_true_, y_pred_, average=None)

        logging.info(f'--- Accuracy: {acc}\t IoU: {jac} ---')
        logging.info(f'--- IoU by Class: {jac_all} ---')
        
        report = metrics.classification_report(y_true_, y_pred_, target_names=self.classes)
        logging.info(f'--- Relatório ---\n{report}\n')

    def _report(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN) if (TP+FN)!=0 else 0
        TNR = TN/(TN+FP) if (TN+FP)!=0 else 0
        PPV = TP/(TP+FP) if (TP+FP)!=0 else 0
        report = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 
                'TPR': TPR, 'Recall': TPR, 'Sensitivity': TPR,
                'TNR' : TNR, 'Specificity': TNR,
                'FPR': FP/(FP+TN) if (FP+TN)!=0 else 0,
                'FNR': FN/(FN+TP) if (FN+TP)!=0 else 0,
                'PPV': PPV, 'Precision': PPV,
                'F1 Score': 2*(PPV*TPR)/(PPV+TPR)
                }
        return report

if __name__ == '__main__':
    classes = ["Urbano", "Mata", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"]
    test_result_path = 'D:\\Projetos\\icmbio\\tmp\\20230302_cross_entropy_100_epoch_weights_unetplusplus\\segnet256_test_result.npz'

    TestTrained(test_result_path, classes).run()