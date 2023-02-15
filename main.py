from skimage import io
import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from utils import load_loss_weights, batch_mean_and_sd

from dataset import DatasetIcmbio
from trainer import Trainer
from trainer_segformer import TrainerSegformer
from models import SegNet, SegNet_two_pools_test
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from utils import clear, convert_to_color, make_optimizer, seed_everything, visualize_augmentations
from sklearn.model_selection import train_test_split
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_preprocessing_fn
# from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch


def is_save_epoch(epoch, ignore_epoch=0):
    return params['save_epoch'] is not None and epoch % params['save_epoch'] == 0 and epoch != ignore_epoch

if __name__=='__main__':
    
    # Params
    params = {
        'root_dir': 'D:\\datasets\\ICMBIO_NOVO\\all',
        'window_size': (256, 256),
        'cache': True,
        'bs': 8,
        'n_classes': 8,
        'classes': ["Urbano", "Mata", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"],
        'cpu': None,
        'device': 'cuda',
        'precision' : 'full',
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
        'save_epoch': 5,
        'print_each': 500,
    }
    
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
    
    # image_folder = os.path.join(params['root_dir'], 'images/000000{}.tif')
    # label_folder = os.path.join(params['root_dir'], 'label/000000{}.tif')
    # all_files = sorted(glob(label_folder.replace('{}', '*')))
    # all_ids = [os.path.split(f)[1].split('.')[0] for f in all_files]
    
    
    # split train/val (80/20)
    # train_images, val_images = np.split(train_images, [int(len(train_images)*0.8)])
    # train_labels, val_labels = np.split(train_labels, [int(len(train_labels)*0.8)])
    # train_images, val_images = train_images.tolist(), val_images.tolist()
    # train_labels, val_labels = train_labels.tolist(), val_labels.tolist()
    
    train_transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=1),
        # A.RandomBrightnessContrast(p=0.2),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
        # A.ChannelShuffle(p=1),
        # A.ColorJitter(brightness=0,contrast=(1,5),saturation=0,hue=0),
        # A.RGBShift(r_shift_limit=(0,255),g_shift_limit=(-40,40),b_shift_limit=(-30,30),p=0.5)
        # A.Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5],
        # ),
        ToTensorV2()
    ])

    # Create train and test sets
    train_dataset = DatasetIcmbio(train_images, train_labels, window_size = params['window_size'], cache = params['cache'], transform=train_transform)
    test_dataset = DatasetIcmbio(test_images, test_labels, window_size = params['window_size'], cache = params['cache'])

    # # Load dataset classes in pytorch dataloader handler object
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['bs'], num_workers=0, pin_memory=False, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params['bs'], num_workers=0, pin_memory=False, shuffle=False)

    # mean, std = batch_mean_and_sd(train_loader)
    
    # # Load network model in cuda (gpu)
    # model = SegNet(in_channels = 3, out_channels = params['n_classes'])
    # model = SegNet_two_pools_test(in_channels = 3, out_channels = params['n_classes'], pretrained = True, pool_type = 'dwt')
    # model.cuda()

    df = pd.read_csv('classes.csv')
    classes = df['name']
    palette = df[[' r', ' g', ' b']].values
    id2label = classes.to_dict()
    label2id = {v: k for k, v in id2label.items()}
    num_labels=len(label2id)
    params['id2label'] = id2label
    params['palette'] = palette

    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     'nvidia/segformer-b0-finetuned-ade-512-512',
    #     num_labels=params['n_classes'],
    #     id2label=id2label,
    #     label2id=label2id,
    #     ignore_mismatched_sizes=True,
    # )
    # model.cuda()
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True,
            num_labels=len(id2label), id2label=id2label, label2id=label2id,
            reshape_last_stage=True)
    model.to(params['device'])

    loader = {
        "train": train_loader,
        "test": test_loader,
    }
    
    cbkp=None
    trainer = TrainerSegformer(model, loader, params, cbkp=cbkp)
    # print(trainer.test(stride = 32, all = False))
    # _, all_preds, all_gts = trainer.test(all=True, stride=32)
    clear()
    
    for epoch in range(trainer.last_epoch+1, params['maximum_epochs']):
        trainer.train()
        
        if is_save_epoch(epoch, ignore_epoch=params['maximum_epochs']):
            # acc = trainer.test(stride = min(params['window_size']), all=False)
            trainer.save('./segnet256_epoch_{}.pth.tar'.format(epoch))
            
    trainer.save('./segnet_final_{}.pth.tar'.format(params['maximum_epochs']))
    # acc, all_preds, all_gts = trainer.test(all=True, stride=min(params['window_size']))
    acc, all_preds, all_gts = trainer.test(all=True, stride=32)
    print(f'Global Accuracy: {acc}')
    
    input_ids, label_ids = test_loader.dataset.get_dataset()
    all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
    
    for p, id_ in zip(all_preds, all_ids):
        img = convert_to_color(p)
        # plt.imshow(img) and plt.show()
        io.imsave('./tmp/inference_tile_{}.png'.format(id_), img)
         