import os
import pickle
import tarfile
import tempfile
from urllib.request import urlretrieve

import numpy as np


def download_cifar100_npy():
    if not (os.path.exists('cifar100_trX.npy') and
            os.path.exists('cifar100_trY.npy') and
            os.path.exists('cifar100_tsX.npy') and
            os.path.exists('cifar100_tsY.npy')):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        with tempfile.NamedTemporaryFile() as tf:
            path, _ = urlretrieve(url, filename=tf.name)
            with tempfile.TemporaryDirectory() as td:
                with tarfile.open(path) as tar:
                    tar.extractall(path=td)
                for fname in ['train', 'test']:
                    with open(os.path.join(td, 'cifar-100-python', fname), 'rb') as f:
                        d = pickle.load(f, encoding='bytes')
                    for k, v in d.items():
                        key = k.decode('utf8')
                        if key == 'data':
                            x = v.reshape(v.shape[0], 3, 32, 32)
                            np.save('cifar100_' + ('tr' if fname == 'train' else 'ts') + 'X.npy', x)
                        elif key == 'fine_labels':
                            y = np.array(v).reshape(len(v), 1)
                            one_hot = np.zeros((y.size, y.max() + 1))
                            one_hot[np.arange(y.size), y] = 1
                            np.save('cifar100_' + ('tr' if fname == 'train' else 'ts') + 'Y.npy', one_hot)
