import os.path as osp
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset


def get_data_dir(identifier):
    base_dir = osp.join(osp.dirname(__file__), 'data')
    if identifier.startswith('parity'):
        if identifier == 'parity':
            return osp.join(base_dir, 'parity', '40')
        else:
            assert identifier.startswith('parity-')
            return osp.join(base_dir, 'parity', identifier[7:])
    elif identifier == 'sudoku-rrn':
        return osp.join(base_dir, 'sudoku-rrn')
    elif identifier.startswith('sudoku'):
        return osp.join(base_dir, 'sudoku')
    else:
        raise ValueError('Unknown dataset: {}.'.format(identifier))


def load_satnet_dataset(data_dir):
    if not osp.exists(data_dir):
        raise ValueError(f'Data directory {data_dir} does not exist. Run data/download-satnet.sh to download the dataset.')
    features = torch.load(osp.join(data_dir, 'features.pt'))
    labels = torch.load(osp.join(data_dir, 'labels.pt'))
    return features, labels


def load_rrn_dataset(data_dir, split):
    if not osp.exists(data_dir):
        raise ValueError(f'Data directory {data_dir} does not exist. Run data/download-rrn.sh to download the dataset.')

    split_to_filename = {
        'train': 'train.csv',
        'val': 'valid.csv',
        'test': 'test.csv'
    }

    filename = osp.join(data_dir, split_to_filename[split])
    df = pd.read_csv(filename, header=None)

    def str2onehot(x):
        x = np.array(list(map(int, x)), dtype='int64')
        y = np.zeros((len(x), 9), dtype='float32')
        idx = np.where(x > 0)[0]
        y[idx, x[idx] - 1] = 1
        return y.reshape((9, 9, 9))

    features = list()
    labels = list()
    for i in range(len(df)):
        inp = df.iloc[i][0]
        out = df.iloc[i][1]
        features.append(str2onehot(inp))
        labels.append(str2onehot(out))

    return torch.tensor(np.array(features)), torch.tensor(np.array(labels))


class SATNetDataset(Dataset):
    def __init__(self, dataset_identifier):
        self.features, self.labels = load_satnet_dataset(get_data_dir(dataset_identifier))
        self.inp_dim = self.features[0].numel()
        self.out_dim = self.labels[0].numel()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return _rescale(self.features[idx].reshape(-1)), _rescale(self.labels[idx].reshape(-1))


class SudokuDataset(Dataset):
    def __init__(self, dataset_identifier, split):
        self.features, self.labels = load_satnet_dataset(get_data_dir(dataset_identifier))
        nr_datapoints = len(self.features)

        assert split in ('train', 'val')
        self.split = split
        if self.split == 'train':
            self.features = self.features[:int(nr_datapoints * 0.9)]
            self.labels = self.labels[:int(nr_datapoints * 0.9)]
        else:
            self.features = self.features[int(nr_datapoints * 0.9):]
            self.labels = self.labels[int(nr_datapoints * 0.9):]

        self.cond_entry = (self.features.sum(axis=-1) == 1)[:, :, :, None].expand(-1, -1, -1, 9)
        self.inp_dim = self.features[0].numel()
        self.out_dim = self.labels[0].numel()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return _rescale(self.features[idx].reshape(-1)), _rescale(self.labels[idx].reshape(-1)), self.cond_entry[idx].reshape(-1)


def _rescale(x):
    return (x - 0.5) * 2


class SudokuRRNDataset(Dataset):
    def __init__(self, dataset_identifier, split):
        assert dataset_identifier == 'sudoku-rrn'
        self.features, self.labels = load_rrn_dataset(get_data_dir(dataset_identifier), split)

        self.cond_entry = (self.features.sum(axis=-1) == 1)[:, :, :, None].expand(-1, -1, -1, 9)
        self.inp_dim = self.features[0].numel()
        self.out_dim = self.labels[0].numel()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return _rescale(self.features[idx].reshape(-1)), _rescale(self.labels[idx].reshape(-1)), self.cond_entry[idx].reshape(-1)


class SudokuRRNLatentDataset(Dataset):
    def __init__(self, dataset_identifier, split):
        data = np.load("data/sudoku-rrn_{}.npz".format(split))
        latent = data['latent']
        inp = data['inp']
        mask = data['mask']
        label = data['label']

        self.latent = latent
        self.inp = inp
        self.mask = mask
        self.label = label

        self.norm = 4

        self.latent = self.latent / self.norm

        self.inp_dim = 729
        self.out_dim = 243

    def __len__(self):
        return self.latent.shape[0]

    def __getitem__(self, idx):
        inp = self.inp[idx]
        latent = self.latent[idx]
        mask = self.mask[idx]
        label = self.label[idx]

        latent = latent.transpose((1, 2, 0)).reshape(-1)

        return inp, latent, label, mask
