import numpy as np

# from utils import plot_confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


if __name__ == '__main__':
    # plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
    #                                             [  604,  4392,  6233],
    #                                             [  162,  2362, 31760]]), 
    #                     normalize    = True,
    #                     target_names = ['high', 'medium', 'low'],
    #                     title        = "Confusion Matrix")
    
    multiclass = np.array([[2, 1, 0, 0],
                       [1, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    class_names = ['class a', 'class b', 'class c', 'class d']

    # fig = plt.figure(figsize=(8, 6))
    fig, ax = plot_confusion_matrix(    conf_mat=multiclass,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=class_names)
    # plt.show()
    fig.savefig(f"./tmp/cm", dpi=fig.dpi, bbox_inches='tight')