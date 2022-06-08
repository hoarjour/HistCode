import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

class Gene_Reg_Dataset(Dataset):
    def __init__(self, h5_dir, csv_path, test=False):
        self.h5_dir = h5_dir
        self.csv_path = csv_path
        self.train_data = pd.read_csv(csv_path, index_col="slide_id")
        self.test = test

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        h5_file = self.train_data.index[idx]
        h5_file = h5_file.split(".svs")[0] + ".h5"
        h5_path = os.path.join(self.h5_dir, h5_file)

        # read h5 file
        f = h5py.File(h5_path, 'r')
        features = f["features"][()]
        f.close()

        features_ = torch.from_numpy(features)

        # read label
        labels = self.train_data.iloc[idx].values
        labels = np.expand_dims(labels, axis=0)
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)
        if self.test:
            return features_, h5_file
        else:
            return features_, labels


class Simple_Dataset(Dataset):

    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.feats = os.listdir(feat_dir)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        # read h5 file
        h5_name = self.feats[idx]
        h5_path = os.path.join(self.feat_dir, h5_name)
        f = h5py.File(h5_path, 'r')
        features = f["features"][()]
        f.close()

        features_ = torch.from_numpy(features)

        return features_, h5_name


class Multi_Gene_Reg_Dataset(Dataset):

    def __init__(self, h5_dir, csv_path, test=False, norm=True):
        self.h5_dir = h5_dir
        self.csv_path = csv_path
        self.train_data = pd.read_csv(csv_path)
        self.test = test
        self.gene_name = None

        self.features = []

        # read all h5 feature file
        for slide_id in self.train_data["slide_id"].values:
            h5_file = slide_id.split(".svs")[0] + ".h5"
            h5_path = os.path.join(self.h5_dir, h5_file)

            f = h5py.File(h5_path, 'r')
            features = f["features"][()]
            f.close()
            features = torch.from_numpy(features)

            self.features.append(features)
        print(f"all feature files are read, length: {len(self.features)}")

        # log10(1 + n) normalization
        if norm:
            self._norm()

    def switch(self, gene_name):
        self.gene_name = gene_name

    def get_all_gene_names(self):
        return self.train_data.columns[1:]

    def __len__(self):
        return self.train_data.shape[0]

    def _norm(self):
        columns = self.train_data.columns[1:]
        # log10(n + 1)
        for col in columns.values:
            self.train_data[col] = np.log10(self.train_data[col].values + 1)

    def __getitem__(self, idx):
        if idx == 0:
            if self.test:
                print(f"use gene dataset {self.gene_name} for testing")
            else:
                print(f"use gene dataset  {self.gene_name} for training")
        labels = self.train_data[self.gene_name][idx]
        labels = np.expand_dims(labels, axis=0)
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)
        return self.features[idx], labels

class Multi_Gene_Clf_Dataset(Dataset):

    def __init__(self, h5_dir, csv_path, test=False):
        self.h5_dir = h5_dir
        self.csv_path = csv_path
        self.train_data = pd.read_csv(csv_path)
        self.test = test
        self.gene_name = None

        self.features = []

        # read all h5 feature file
        for slide_id in self.train_data["slide_id"].values:
            h5_file = slide_id.split(".svs")[0] + ".h5"
            h5_path = os.path.join(self.h5_dir, h5_file)

            # 读取h5格式的特征文件
            f = h5py.File(h5_path, 'r')
            features = f["features"][()]
            f.close()
            features = torch.from_numpy(features)

            self.features.append(features)
        print(f"all feature files are read, length: {len(self.features)}")

    def switch(self, gene_name):
        self.gene_name = gene_name

    def get_all_gene_names(self):
        return self.train_data.columns[1:]

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        if idx == 0:
            if self.test:
                print(f"use gene dataset {self.gene_name} for testing")
            else:
                print(f"use gene dataset  {self.gene_name} for training")
        label = self.train_data[self.gene_name][idx]

        return self.features[idx], label