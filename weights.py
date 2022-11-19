import os
import glob
from skimage import io
from utils import convert_from_color
import numpy as np

EXT = 'tif'

class Weights():
    
    def __init__(self, label_dir: str, classes: list, dev: bool):
        self.label_dir = label_dir
        self.classes = classes
        
        assert os.path.exists(label_dir), "{}. Diretório não existe".format(label_dir)
        
        image_list = glob.glob(f"{label_dir}/*.{EXT}")
        if dev:
            self.images = [io.imread(file) for file in image_list[0:2]]
        else:
            self.images = [io.imread(file) for file in image_list]
        
    def calcular(self) -> list:
        image_color = [convert_from_color(image[:,:,:3]) for image in self.images]
        
        pixels_by_class = []
        for image in image_color:
            gen = ([image.ravel() == i for i in np.arange(9)])
            pixels_by_class.append(np.sum(list(gen), axis=1))
            
        print(pixels_by_class)
            
        # pixel_total = [np.sum([image == i for i in np.arange(9)]) for image in image_color]
        # pixels_by_class = np.sum(list(gen), axis=1)
        
        
if __name__=='__main__':
    
    weights = Weights(
        label_dir='D:\\datasets\\ICMBIO\\all\\label',
        classes=["Urbano", "Mata", "Piscina", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"],
        dev=True
    )
    w = weights.calcular()