__author__ = 'shaozc'

import torch.nn.functional as F
import numpy as np 
import torch.nn as nn
from .focal_loss import FocalLoss # Added this import

# Hardcoded class weights for Fold 0 based on training set distribution
# Class 0 (Majority): 188 samples, Weight: 0.5505
# Class 1 (Minority): 19 samples, Weight: 5.4474
# Calculated as N / (C * N_c) where N=total, C=num_classes, N_c=class_count
class_weights_fold0 = np.array([0.5505, 5.4474], dtype=np.float32)

def create_loss(args, w1=1.0, w2=0.5):
    conf_loss = args['base_loss']
    ### MulticlassJaccardLoss(classes=np.arange(11)
    # mode = args.base_loss #BINARY_MODE \MULTICLASS_MODE \MULTILABEL_MODE 
    loss = None
    if hasattr(nn, conf_loss): 
        loss = getattr(nn, conf_loss)() 
    #binary loss
    elif conf_loss == "focal":
        loss = FocalLoss(alpha=class_weights_fold0, apply_nonlin=lambda x: F.softmax(x, dim=1))
    elif conf_loss == "jaccard":
        loss = L.BinaryJaccardLoss()
    elif conf_loss == "jaccard_log":
        loss = L.BinaryJaccardLoss()
    elif conf_loss == "dice":
        loss = L.BinaryDiceLoss()
    elif conf_loss == "dice_log":
        loss = L.BinaryDiceLogLoss()
    elif conf_loss == "dice_log":
        loss = L.BinaryDiceLogLoss()
    elif conf_loss == "bce+lovasz":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryLovaszLoss(), w1, w2)
    elif conf_loss == "lovasz":
        loss = L.BinaryLovaszLoss()
    elif conf_loss == "bce+jaccard":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryJaccardLoss(), w1, w2)
    elif conf_loss == "bce+log_jaccard":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryJaccardLogLoss(), w1, w2)
    elif conf_loss == "bce+log_dice":
        loss = L.JointLoss(BCEWithLogitsLoss(), L.BinaryDiceLogLoss(), w1, w2)
    elif conf_loss == "reduced_focal":
        loss = L.BinaryFocalLoss(reduced=True)
    else:
        assert False and "Invalid loss"
        raise ValueError
    return loss

import argparse
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-loss', default='CrossEntropyLoss',type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = make_parse()
    myloss = create_loss(args)
    data = torch.randn(2, 3)
    label = torch.empty(2, dtype=torch.long).random_(3)
    loss = myloss(data, label)