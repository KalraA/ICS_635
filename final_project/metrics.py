import os
import numpy as np
import torch
import pickle

def iou(x, y, num_classes=21):
    if x.shape[1] > 1 and len(x.shape) == 4:
        x = torch.argmax(x, axis=1)
    if len(x.shape) == 4:
        x = x[:, 0]

    b, h, w = x.shape
    x = x.reshape(b, -1)
    y = y.reshape(b, -1)
    
    per_class_scores = torch.zeros((b, num_classes), device=x.device)
    per_gt_sum = torch.zeros((b, num_classes), device=x.device)
    for i in range(num_classes):
        mask_pred = x == i 
        mask_gt = y == i
        gt_sum = mask_gt.sum(dim=1)
        pred_sum = mask_pred.sum(dim=1)
        inter = (mask_gt*mask_pred).sum(dim=1)
        union = gt_sum + pred_sum - inter
        iou = inter / (union + 1e-4)
        per_class_scores[:, i] = iou
        per_gt_sum[:, i] = gt_sum
    return per_class_scores, per_gt_sum

class Metrics:
    def __init__(self, name):
        self.name = name
        self.iou_acc = []
        self.iou_denom = []
        self.loss_acc = []
        self.loss = []
        self.miou = []

    def end_iter(self, ious, denoms, loss):
        self.iou_acc.append(ious.cpu().numpy())
        self.iou_denom.append(denoms.cpu().numpy())
        self.loss_acc.append(loss.detach().cpu().numpy())

    def end_epoch(self):
        self.loss.append(np.mean(self.loss_acc))
        numers = np.array(self.iou_acc).sum(axis=0)
        denoms = np.array(self.iou_denom).sum(axis=0)
        self.miou.append(numers / (denoms + 1e-5))
        self.iou_acc = []
        self.iou_denom = []
        self.loss_acc = []
        print(self.miou, self.loss)

    def dump(self, path):
        pickle.dump([self.miou, self.loss], open(os.path.join(path, self.name + '_metrics.pkl'), 'wb'))
