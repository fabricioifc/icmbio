from skimage import io
import os, time
import torch
import numpy as np
import pandas as pd
from glob import glob
from utils import load_loss_weights, batch_mean_and_sd

from dataset import DatasetIcmbio
from trainer import Trainer
from trainer_segformer import TrainerSegformer
from models import build_model
import matplotlib.pyplot as plt
from utils import clear, convert_to_color, make_optimizer, seed_everything, visualize_augmentations
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def is_save_epoch(epoch, ignore_epoch=0):
    return params['save_epoch'] is not None and epoch % params['save_epoch'] == 0 and epoch != ignore_epoch

if __name__=='__main__':
    
    # Params
    params = {
        'results_folder': 'tmp\\20230303_cross_entropy_100_epoch_weights_deeplabv3',
        'root_dir': 'D:\\datasets\\ICMBIO_NOVO\\all',
        'window_size': (256, 256),
        'cache': True,
        'bs': 16,
        'n_classes': 8,
        'classes': ["Urbano", "Mata", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"],
        'cpu': None,
        'device': 'cuda',
        'precision' : 'full',
        'model': {
            'name': 'unet',
            'pretrained': True
        },
        'optimizer_params': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005
        },
        'lrs_params': {
            'type': 'multi',
            'milestones': [25, 35, 45],
            'gamma': 0.1
        },
        'weights': '',
        'maximum_epochs': 100,
        'save_epoch': 10,
        'print_each': 500,
    }
    
    # Carregar os pesos de cada classe, calculados pelo arquivo `extra\weights_calcupator.py`
    params['weights'] = torch.ones(params['n_classes'])
    loss_weights = load_loss_weights('./loss_weights.npy')
    if loss_weights is not None:
        params['weights'] = torch.from_numpy(loss_weights['weights']).float()

    image_dir = os.path.join(params['root_dir'], 'images')
    label_dir = os.path.join(params['root_dir'], 'label')

    # Load image and label files from .txt
    train_images = pd.read_table('train_images.txt',header=None).values
    train_images = [os.path.join(image_dir, f[0]) for f in train_images]
    train_labels = pd.read_table('train_labels.txt',header=None).values
    train_labels = [os.path.join(label_dir, f[0]) for f in train_labels]
    
    test_images = pd.read_table('test_images.txt',header=None).values
    test_images = [os.path.join(image_dir, f[0]) for f in test_images]
    test_labels = pd.read_table('test_labels.txt',header=None).values
    test_labels = [os.path.join(label_dir, f[0]) for f in test_labels]

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),              
        ToTensorV2(),
    ])

    # Create train and test sets
    train_dataset = DatasetIcmbio(train_images, train_labels, window_size = params['window_size'], cache = params['cache'], transform=train_transform)
    test_dataset = DatasetIcmbio(test_images, test_labels, window_size = params['window_size'], cache = params['cache'])

    # # Load dataset classes in pytorch dataloader handler object
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['bs'], num_workers=0, pin_memory=False, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params['bs'], num_workers=0, pin_memory=False, shuffle=False)

    model = build_model(model_name=params['model']['name'], params=params)

    loader = {
        "train": train_loader,
        "test": test_loader,
    }
    
    cbkp=None#'tmp\\20230223_cross_entropy_100_epoch_weights_unetplusplus\\unetplusplus_50.pth.tar'
    trainer = Trainer(model, loader, params, cbkp=cbkp)
    # print(trainer.test(stride = 32, all = False))
    # _, all_preds, all_gts = trainer.test(all=True, stride=32)
    clear()
    
    # Start the training.
    for epoch in range(trainer.last_epoch+1, params['maximum_epochs']):
        trainer.train()
        
        if is_save_epoch(epoch, ignore_epoch=params['maximum_epochs']):
            # acc = trainer.test(stride = min(params['window_size']), all=False)
            # trainer.save('./segnet256_epoch_{}.pth.tar'.format(epoch))
            trainer.save(os.path.join(params['results_folder'], 'trained_epoch_{}.pth.tar'.format(epoch)))
            
    trainer.save(os.path.join(params['results_folder'], '{}_{}.pth.tar'.format(params['model']['name'], params['maximum_epochs'])))
    # acc, all_preds, all_gts = trainer.test(all=True, stride=min(params['window_size']))
    acc, all_preds, all_gts = trainer.test(all=True, stride=32)
    print(f'Global Accuracy: {acc}')
    
    input_ids, label_ids = test_loader.dataset.get_dataset()
    all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
    
    for p, id_ in zip(all_preds, all_ids):
        img = convert_to_color(p)
        # plt.imshow(img) and plt.show()
        # io.imsave('./tmp/inference_tile_{}.png'.format(id_), img)
        io.imsave(os.path.join(params['results_folder'], 'inference', 'inference_tile_{}.png'.format(id_)), img)
         