import custom_transforms as CT
import transforms as T
import torchvision.datasets as datasets
import numpy as np
import torch

VOC_COLORMAP = {
    0: [0, 0, 0],        # Background
    1: [128, 0, 0],      # Aeroplane
    2: [0, 128, 0],      # Bicycle
    3: [128, 128, 0],    # Bird
    4: [0, 0, 128],      # Boat
    5: [128, 0, 128],    # Bottle
    6: [0, 128, 128],    # Bus
    7: [128, 128, 128],  # Car
    8: [64, 0, 0],       # Cat
    9: [192, 0, 0],      # Chair
    10: [64, 128, 0],     # Cow
    11: [192, 128, 0],    # Dining Table
    12: [64, 0, 128],     # Dog
    13: [192, 0, 128],    # Horse
    14: [64, 128, 128],   # Motorbike
    15: [192, 128, 128],  # Person
    16: [0, 64, 0],       # Potted Plant
    17: [128, 64, 0],     # Sheep
    18: [0, 192, 0],      # Sofa
    19: [128, 192, 0],    # Train
    20: [0, 64, 128]     # TV/Monitor
}

def get_voc_data(download=True, add2012=True, photometric_augs=False, geometric_augs=False):
    trans = T.Compose([CT.Load(VOC_COLORMAP),
                        T.RandomResize(192), 
                        T.CenterCrop(392), 
                        CT.DinoPostProcess('facebook/dinov2-small')])
    
    train_trans = trans
    if geometric_augs:
        train_trans = T.Compose([CT.Load(VOC_COLORMAP),
                                 T.RandomResize(130, 400), 
                                 T.RandomCrop(392), 
                                 T.RandomHorizontalFlip(0.5),
                                 CT.DinoPostProcess('facebook/dinov2-small')])

    train_seg_dataset = datasets.VOCSegmentation(
        root='data/',
        year='2007',
        image_set='train',  # Using 'train' for segmentation
        download=False,
        transform=None,
        target_transform=None,
        transforms=train_trans,
    )
    if add2012:
        train_seg_dataset_2012 = datasets.VOCSegmentation(
            root='data/',
            year='2012',
            image_set='train',  # Using 'train' for segmentation
            download=download,
            transform=None,
            target_transform=None,
            transforms=train_trans,
        )
        train_seg_dataset = torch.utils.data.ConcatDataset([train_seg_dataset, train_seg_dataset_2012])

    val_seg_dataset = datasets.VOCSegmentation(
        root='data/',
        year='2007',
        image_set='val',  # Using 'train' for segmentation
        download=download,
        transform=None,
        target_transform=None,
        transforms=trans
    )
    test_seg_dataset = datasets.VOCSegmentation(
        root='data/',
        year='2007',
        image_set='test',  # Using 'train' for segmentation
        download=download,
        transform=None,
        target_transform=None,
        transforms=trans
    )
    return train_seg_dataset, val_seg_dataset, test_seg_dataset