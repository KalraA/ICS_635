import os
import time
import json
import torch
import pickle
import metrics
import torch.nn as nn
import torch.optim as optim
import model_builder
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import matplotlib.pyplot as plt
import transformers
import dataset

# CONFIG
config = dict(
    batch_size = 24,
    model_type = 'dpt',
    add_2012 = True,
    photometric_augs = False,
    geometric_augs = True,
    da_weights = True,
    backbone_distance_coeff = 0,
    weight=0.1,
    epochs=350,
    lr=0.0003,
)


log_folder = os.path.join('logs', str(int(time.time())))
os.makedirs(log_folder, exist_ok=True)
json.dump(config, open(os.path.join(log_folder, 'config.json'), 'w'))

train_seg_dataset, val_seg_dataset, test_seg_dataset = dataset.get_voc_data(download=False, 
                                                                            add2012=config['add_2012'], 
                                                                            photometric_augs=config['photometric_augs'], 
                                                                            geometric_augs=config['geometric_augs'])
train_loader = DataLoader(train_seg_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_seg_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)
test_loader = DataLoader(test_seg_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)
# Define a simple segmentation model (using a pre-trained ResNet backbone)


# Get the number of classes in the VOC 2007 dataset (including background)
num_classes = len(dataset.VOC_COLORMAP)

# Losses

if config['weight'] == 'class':
    weights = torch.from_numpy(np.array([0.010962754774204456, 6.188964772990057, 7.275230155853571, 3.411490970894412, 6.9654881037586485, 4.668329965840541, 1.4887161610917452, 4.344901780399372, 1.7328032804575375, 2.6630879962851175, 4.097497543984996, 2.5015014098110386, 2.224007201966691, 3.6357207870129282, 3.8957963050571953, 0.3522830218275276, 9.924222366428724, 14.739444673918033, 1.8012617220801364, 2.5965151025160083, 2.4696084402202327])).cuda().float()
else:
    weights = torch.ones(num_classes).cuda() # Initialize all weights to 1.0
    weights[0] = config['weight'] # Set a lower weight for class 2

criterion = nn.CrossEntropyLoss(weight=weights)


# Initialize the model
warmup_lr = 1e-7
final_lr = config['lr']
warmup_iterations = 113

model = model_builder.build(config['model_type'], num_classes)
if config['model_type'] == 'dpt':
    param_groups = [
        {'params': model.bb.parameters(), 'lr': final_lr / 100},
        {'params': list(model.head.parameters()) + list(model.upwards.parameters())  + list(model.and_.parameters()) + list(model.onwards.parameters())}
    ]
elif config['model_type'] == 'simple':
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': final_lr / 100},
        {'params': list(model.conv1.parameters()) + list(model.conv2.parameters())}
    ]

if config['da_weights']:
    state_dict = torch.load('da_weights.pth')
    state_dict.pop('head.conv3.weight')
    state_dict.pop('head.conv3.bias')
    extras, missed = model.load_state_dict(state_dict, strict=False)
    print(len(extras), len(missed))
# Define loss function and optimizer
optimizer = optim.Adam(param_groups, lr=final_lr)

# Check if CUDA is available and move the model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
# train_loader = val_loader

train_metrics = metrics.Metrics('train')
val_metrics = metrics.Metrics('val')
test_metrics = metrics.Metrics('test')
losses = []
num_epochs = config['epochs']
for epoch in range(num_epochs):
    train_metrics.dump(log_folder)
    val_metrics.dump(log_folder)
    test_metrics.dump(log_folder)
    model.train()
    running_loss = 0.0
    t = time.time()
    optimizer.zero_grad()

    for i, data in enumerate(train_loader, 0):
        # if epoch == 0:
            # if i <= warmup_iterations:
            #     # Linear warmup
            #     lr_scale = min(1.0, float(i) / warmup_iterations)
            #     # Option 1: If you calculated initial_warmup_lr based on a factor of target_lr
            #     # current_lr = target_lr * lr_scale
            #     # Option 2: Linear increase from initial_warmup_lr to target_lr
            #     current_lr = warmup_lr + (final_lr - warmup_lr) * lr_scale

            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = current_lr

        print(1, time.time() - t)
        optimizer.zero_grad()
        inputs, masks = data
        inputs = inputs.to(device).float()
        masks = masks.squeeze(1).long().to(device) # Squeeze the channel dimension and convert to long
        outputs = model(inputs)
        with torch.no_grad():
            per_class_scores, per_gt_sum = metrics.iou(outputs, masks)
        loss = criterion(outputs, masks)
        loss.backward()
        losses.append(loss.detach().cpu().numpy())
        running_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(loss)
        if (i + 1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1}/{len(train_loader)}] Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
        train_metrics.end_iter(per_class_scores.sum(axis=0), (per_gt_sum > 1).sum(axis=0), loss)
        t = time.time()
    train_metrics.end_epoch()

    # Validation loop 
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, masks = data
            inputs = inputs.to(device)
            masks = masks.squeeze(1).long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            with torch.no_grad():
                per_class_scores, per_gt_sum = metrics.iou(outputs, masks)

            val_loss += loss.item()
            val_metrics.end_iter(per_class_scores.sum(axis=0), (per_gt_sum > 1).sum(axis=0), loss)
        val_metrics.end_epoch()
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}')

    # Test loop ### NOTE We only use it at the best val iter but we store it for all for convenience
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, masks = data
            inputs = inputs.to(device)
            masks = masks.squeeze(1).long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            with torch.no_grad():
                per_class_scores, per_gt_sum = metrics.iou(outputs, masks)

            test_loss += loss.item()
            test_metrics.end_iter(per_class_scores.sum(axis=0), (per_gt_sum > 1).sum(axis=0), loss)
        test_metrics.end_epoch()
        print(f'Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.4f}')
    torch.save(model.state_dict(), os.path.join(log_folder, 'model.pth'))


print('Finished Training')