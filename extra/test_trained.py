import sys
sys.path.append('D:\\Projetos\\icmbio\\')
import matplotlib
matplotlib._log.disabled = True

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
    
    def run(self):
        logging.info('\n\n--- Iniciando o teste para gerar relatórios ---')
        logging.info(f'--- {test_result_path} ---')

        acc = self.test.f.arr_0.item().get('acc')
        logging.info(f'--- Accuracy: {acc} ---')

        logging.info(f'--- Carregando o arquivo com as previsões e ground truths ---')
        y_pred = self.test.f.arr_0.item().get('all_preds')
        y_true = self.test.f.arr_0.item().get('all_gts')

        logging.info(f'--- Geranco a Matriz de Confusão ---')
        cm = confusion_matrix(
            np.concatenate([p.ravel() for p in y_true]),
            np.concatenate([p.ravel() for p in y_pred]),
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

        logging.info(f'--- Convertendo Array para Tensor ---')
        y_true_ = torch.from_numpy(np.concatenate([p.ravel() for p in y_true]))
        y_pred_ = torch.from_numpy(np.concatenate([p.ravel() for p in y_pred]))

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

        
    # def run_test(self):
    #     acc, all_preds, all_gts = self.trainer.test(all=True, stride=self.stride)
    #     print(f'Global Accuracy: {acc}')
        
    #     input_ids, label_ids = self.trainer.loader.dataset.get_dataset()
    #     all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
        
    #     for p, id_ in zip(all_preds, all_ids):
    #         img = convert_to_color(p)
    #         # plt.imshow(img) and plt.show()
    #         io.imsave(os.path.join(RESULTS_PATH, f'inference_tile_{id_}.png'), img)

if __name__ == '__main__':
    classes = ["Urbano", "Mata", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"]
    test_result_path = 'D:\\Projetos\\icmbio\\tmp\\20230208_cross_entropy_100_epoch_weights_1\\segnet256_test_result.npz'
    

    TestTrained(test_result_path, classes).run()
    
127418

# def __init__(self, trainer: Trainer, stride=None):
#     self.trainer = trainer
#     self.stride = stride or min(trainer.params['window_size'])
#     Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    
# def run_test(self):
#     acc, all_preds, all_gts = self.trainer.test(all=True, stride=self.stride)
#     print(f'Global Accuracy: {acc}')
    
#     input_ids, label_ids = self.trainer.loader.dataset.get_dataset()
#     all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
    
#     for p, id_ in zip(all_preds, all_ids):
#         img = convert_to_color(p)
#         # plt.imshow(img) and plt.show()
#         io.imsave(os.path.join(RESULTS_PATH, f'inference_tile_{id_}.png'), img)