import os
import glob
from skimage import io
from utils import convert_from_color
import numpy as np

EXT = 'tif'

class WeightsCalculator():
    
    def __init__(self, label_dir: str, classes: list, dev=False):
        self.label_dir = label_dir
        self.C = len(classes) # quantidade de classes
        self.N = None # Pixels por classe
        self.M = None # Soma dos pixels de todas as imagens
        
        assert os.path.exists(label_dir), "{}. Diretório não existe".format(label_dir)
        
        image_list = glob.glob(f"{label_dir}/*.{EXT}")

        print(f'--- Carregando as {len(image_list)} imagens ---')
        if dev:
            self.images = [io.imread(file) for file in image_list[0:3]]
        else:
            self.images = [io.imread(file) for file in image_list]
        
        self.load()
        
    '''Calcular os pixels por classe de todas as imagens e os pixels totais'''
    def load(self) -> None:
        print(f'--- Convertendo imagem para Paleta de Cores ---')
        image_color = [convert_from_color(image[:,:,:3]) for image in self.images]
        
        pixels_by_image_class = []
        for idx, image in enumerate(image_color):
            print(f'--- Somando pixels da imagem {idx+1} ---')
            gen = ([image.ravel() == i for i in np.arange(self.C)])
            pixels_by_image_class.append(np.sum(list(gen), axis=1))
            
        self.N = np.sum(pixels_by_image_class, axis=0)
        self.M = np.sum(pixels_by_image_class)
        # print(pixels_by_image_class)
        # print(self.N)
        # print(self.M)
        
    def calcular(self, normalize=True) -> list:
        prod = np.array([self.C*n for n in self.N])
        weights = np.divide(self.M, prod, where=prod!=0)
        if normalize:
            weights = np.divide(weights, np.linalg.norm(weights))
        return weights
        
        
if __name__=='__main__':
    
    wc= WeightsCalculator(
        label_dir='D:\\datasets\\ICMBIO\\all\\label',
        classes=["Urbano", "Mata", "Piscina", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"],
        dev=False
    )
    weights = wc.calcular()
    
    print(weights)
    
    # N = [114042226, 497764795, 1795239, 56643355, 173535660, 21323789, 29305581, 0]
    # C = 9
    # M = 910163968
    
    # pesos = np.divide(M, [C*n for n in N])
    # print(pesos)
    # print(np.divide(pesos, np.linalg.norm(pesos)))
    
    