# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: generate_noisy.py
@Time: 2019/10/19 下午7:46
@Desc: generate_noisy.py
"""


try:
    import os

    import torch
    import pandas as pd
    import numpy as np

    from vagan.config import DATASETS_DIR

except ImportError as e:
    print(e)
    raise ImportError


class Generator:
    """
    generate noisy data
    """
    def __init__(self):

        self.low_range = (0, 10)
        self.upper_range = (245, 255)

    def generate(self, size=70000, dim=784, channels=1, data_name='mnist'):

        noisy_size = int(size * 0.05)
        datas = None
        for i in range(noisy_size):
            data1 = np.random.uniform(self.low_range[0], self.low_range[1], (channels, dim))
            data2 = np.random.uniform(self.upper_range[0], self.upper_range[1], (channels, dim))
            if datas is None:
                datas = data1
                datas = np.vstack((datas, data2))
            else:
                datas = np.vstack((datas, data1))
                datas = np.vstack((datas, data2))
        datas = pd.DataFrame(datas)
        datas.to_csv(os.path.join(DATASETS_DIR, '{}_noisy.csv'.format(data_name)))


if __name__ == "__main__":

    generator = Generator()

    generator.generate(size=70000, dim=784, channels=1, data_name='mnist')
    # generator.generate(size=60000, dim=1024, channels=3, data_name='cifar10')
