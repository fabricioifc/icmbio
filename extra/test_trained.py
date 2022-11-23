import os, io
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

RESULTS_PATH = './results'

class TestTrainedModel:
    
    def __init__(self, trainer: Trainer, stride=None):
        self.trainer = trainer
        self.stride = stride or min(trainer.params['window_size'])
        Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        
    def run(self):
        acc, all_preds, all_gts = self.trainer.test(all=True, stride=self.stride)
        print(f'Global Accuracy: {acc}')
        
        input_ids, label_ids = self.trainer.loader.dataset.get_dataset()
        all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
        
        for p, id_ in zip(all_preds, all_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave(os.path.join(RESULTS_PATH, f'inference_tile_{id_}.png'), img)
        
            
            

if __name__ == '__main__':
    test_result_path = 'D:\\Projetos\\icmbio\\tmp\\20221122_cross_entropy\\segnet256_test_result.npz'
    test = np.load(test_result_path, allow_pickle=True)

    acc = test.f.arr_0.item().get('acc')
    y_pred = test.f.arr_0.item().get('all_preds')
    y_true = test.f.arr_0.item().get('all_gts')
    classes = ["Urbano", "Mata", "Piscina", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"]
    print(acc)
    
    cm = confusion_matrix(
        np.concatenate([p.ravel() for p in y_true]),
        np.concatenate([p.ravel() for p in y_pred]),
        labels=range(len(classes)),
    )
    
    fig, ax = plot_confusion_matrix(
        conf_mat=cm, 
        figsize=(10,10), 
        cmap=plt.cm.Greens,
        colorbar=True,
        show_absolute=False,
        show_normed=True,
    )

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title('Confusion Matrix', fontsize=12)
    plt.show()