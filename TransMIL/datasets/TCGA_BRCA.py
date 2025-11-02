import torch
import pandas as pd
from pathlib import Path
import os

import torch.utils.data as data

class TcgaBrca(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.split_dir = self.dataset_cfg.label_dir

        self.shuffle = self.dataset_cfg.data_shuffle

        if state == 'train':
            split_path = os.path.join(self.split_dir, str(self.fold), 'train.csv')
            self.slide_data = pd.read_csv(split_path)
        elif state == 'val':
            # Using the test set for validation as per the CLAM setup
            split_path = os.path.join(self.split_dir, str(self.fold), 'test.csv')
            self.slide_data = pd.read_csv(split_path)
        elif state == 'test':
            split_path = os.path.join(self.split_dir, str(self.fold), 'test.csv')
            self.slide_data = pd.read_csv(split_path)

        self.data = self.slide_data['slide_id']
        self.label = self.slide_data['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        if not full_path.exists():
            # Fallback for .h5 files if .pt is not found
            full_path = Path(self.feature_dir) / f'{slide_id}.h5'
        
        features = torch.load(full_path)

        if self.shuffle:
            index = list(range(features.shape[0]))
            random.shuffle(index)
            features = features[index]

        return features, label
