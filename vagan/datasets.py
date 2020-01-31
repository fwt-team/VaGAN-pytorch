# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: datasets.py
@time: 2019/6/9 下午8:20
@desc: datasets
"""

try:
    import copy
    import os

    import torch
    import torchvision.datasets as dset
    import numpy as np
    import scipy.io as scio
    import torchvision.transforms as transforms

    from torch.utils.data import Dataset

    from vagan.config import DATASETS_DIR, THRESHOLD
    from vagan.utils import get_one_data_from_noisy

except ImportError as e:
    print(e)
    raise ImportError


class Reuters10k(Dataset):

    def __init__(self, root, train=True, transforms=None):
        super(Reuters10k, self).__init__()

        self.root = root
        self.train = train
        self.transform = transforms

        self.data, self.label = self.load_reuter10k_data(self.root, self.train)

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):

        return len(self.label)

    def load_reuter10k_data(self, root, train=True):

        data = scio.loadmat(os.path.join(root, 'reuters10k.mat'))

        data_len = data['Y'].size
        train_length = int(data_len - np.ceil(data_len*0.1))
        if train:
            X = data['X'][:train_length]
            Y = data['Y'].squeeze()[:train_length]
        else:
            X = data['X'][train_length:]
            Y = data['Y'].squeeze()[train_length:]

        return X, Y.astype(np.int64)


DATASET_FN_DICT = {
    'mnist': dset.MNIST,
    'fashion-mnist': dset.FashionMNIST,
    'cifar10': dset.CIFAR10,
    'reuters10k': Reuters10k,
}


dataset_list = DATASET_FN_DICT.keys()


def _get_dataset(dataset_name='mnist'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


def outliers_transform(image):

    threshold = THRESHOLD
    pro = np.random.uniform()
    if pro <= threshold:
        # draw uniform data to cover image
        image = get_one_data_from_noisy(size=image.size())

    return image


# get the loader of all datas
def get_dataloader(dataset_path='../datasets/mnist',
                   dataset_name='mnist', train=True, batch_size=50, noisy=False):
    dataset = _get_dataset(dataset_name)

    transform = [
        transforms.ToTensor(),
        # transforms.Normalize((0.369,), (0.369))
    ]
    if noisy:
        transform.append(transforms.Lambda(outliers_transform))
    loader = torch.utils.data.DataLoader(
        dataset(dataset_path, download=True, train=train, transform=transforms.Compose(transform)),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader

