import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import os.path as osp
import numpy as np
import json
import torchvision.transforms.functional as TF
import random
from torchvision.datasets import ImageFolder

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob

from glob import glob
# import cv2
# from imageio import imread
# from skimage.transform import resize as imresize

from torchvision.datasets import CIFAR100
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from copy import deepcopy
import time
from scipy.linalg import lu


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class NoisyWrapper:

    def __init__(self, dataset, timesteps):

        self.dataset = dataset
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        self.inp_dim = dataset.inp_dim
        self.out_dim = dataset.out_dim

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod = np.linspace(1, 0, timesteps)
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1. - alphas_cumprod))
        self.extract = extract

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, *args, **kwargs):
        x, y = self.dataset.__getitem__(*args, **kwargs)
        x = torch.tensor(x)
        y = torch.tensor(y)

        t = torch.randint(1, self.timesteps, (1,)).long()
        t_next = t - 1
        noise = torch.randn_like(y)

        sample = (
            self.extract(
                self.sqrt_alphas_cumprod,
                t,
                y.shape) *
            y +
            self.extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                y.shape) *
            noise)

        sample_next = (
            self.extract(
                self.sqrt_alphas_cumprod,
                t_next,
                y.shape) *
            y +
            self.extract(
                self.sqrt_one_minus_alphas_cumprod,
                t_next,
                y.shape) *
            noise)
        return x, sample, sample_next


def conjgrad(A, b, x, num_steps=20):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    sols = [x.flatten()]

    for i in range(num_steps):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + alpha[0, 0] * p
        r = r - alpha[0, 0] * Ap
        rsnew = np.dot(np.transpose(r), r)
        # if np.sqrt(rsnew) < 1e-8:
        #     break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        sols.append(x.flatten())

    sols = np.stack(sols, axis=0)
    return sols


class FiniteWrapper(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, dataset, string, capacity, rank, num_steps):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        cache_file = "cache/{}_{}_{}_{}.npz".format(
            capacity, string, rank, num_steps)
        self.inp_dim = dataset.inp_dim
        self.out_dim = dataset.out_dim

        if osp.exists(cache_file):
            data = np.load(cache_file)
            inps = data['inp']
            outs = data['out']
        else:
            "Generating static dataset for training....."
            inps = []
            outs = []

            for i in tqdm(range(capacity)):
                inp, out, trace = dataset[i]
                inps.append(inp)
                outs.append(out)

            inps = np.array(inps)
            outs = np.array(outs)

            np.savez(cache_file, inp=inps, out=outs)

        self.inp = inps
        self.out = outs

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        inp = self.inp[index]
        out = self.out[index]

        return inp, out

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return self.inp.shape[0]


class Identity(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, num_steps=5):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.num_steps = num_steps

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        R = np.random.uniform(0, 1, (self.h, self.w))
        R_corrupt = R

        R_list = np.tile(R.flatten()[None, :], (self.num_steps, 1))
        R_list = np.concatenate([R_corrupt.flatten()[None, :], R_list], axis=0)

        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class LowRankDataset(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, ood):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w
        self.ood = ood

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        U = np.random.randn(self.h, 10)
        V = np.random.randn(self.w, 10)

        if self.ood:
            R = 0.1 * np.random.randn(self.h, self.w) + np.dot(U, V.T) / 5
        else:
            R = 0.1 * np.random.randn(self.h, self.w) + np.dot(U, V.T) / 20

        mask = np.round(np.random.rand(self.h, self.w))

        R_corrupt = R * mask
        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return int(1e7)


class Negate(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        R = np.random.uniform(0, 1, (self.h, self.w))
        R_corrupt = -1 * R

        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class Addition(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, ood):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank

        if self.h == 2:
            self.w = 1
        else:
            self.w = rank
        self.ood = ood

        self.split = split
        self.inp_dim = 2 * self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        if self.ood:
            scale = 2.5
        else:
            scale = 1.0

        R_one = np.random.uniform(-scale, scale, (self.h, self.w)).flatten()
        R_two = np.random.uniform(-scale, scale, (self.h, self.w)).flatten()
        R_corrupt = np.concatenate([R_one, R_two], axis=0)
        R = R_one + R_two

        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e7)


class Square(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, num_steps=10):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.num_steps = num_steps

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        R_corrupt = np.random.uniform(-1, 1, (self.h, self.w))
        R = np.matmul(R_corrupt, R_corrupt).flatten() / 5.
        # R = R_corrupt * R_corrupt

        R_list = np.tile(R.flatten()[None, :], (self.num_steps, 1))
        R_list = np.concatenate([R_corrupt.flatten()[None, :], R_list], axis=0)

        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class Inverse(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, ood):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.ood = ood

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        R_corrupt = np.random.uniform(-1, 1, (self.h, self.w))
        R_corrupt = R_corrupt.dot(R_corrupt.transpose())
        # R_corrupt = R_corrupt + 0.5 * np.eye(self.h)

        if self.ood:
            R_corrupt = R_corrupt + R_corrupt.transpose() + 0.1 * np.eye(self.h)
        else:
            R_corrupt = R_corrupt + R_corrupt.transpose() + 0.5 * np.eye(self.h)

        R = np.linalg.inv(R_corrupt)
        # R = np.linalg.solve(R_corrupt, np.eye(self.h))

        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e7)


class Equation(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, num_steps=10):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank

        self.split = split
        self.inp_dim = rank * rank + rank
        self.out_dim = rank
        self.num_steps = num_steps

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        A = np.random.uniform(-1, 1, (self.h, self.w))
        A = A + A.transpose()
        A = np.matmul(A, A.transpose())

        x = np.random.uniform(-1, 1, (self.h, 1))
        b = np.matmul(A, x)

        inp = np.concatenate([A.flatten(), b.flatten()], axis=0)
        sol = x.flatten()

        x_guess = np.random.uniform(-1, 1, (self.h, 1))

        return inp, sol

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class LU(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = 2 * self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        R_corrupt = np.random.uniform(-1, 1, (self.h, self.w))
        p, l, u = lu(R_corrupt)
        R = np.concatenate([l.flatten(), u.flatten()], axis=0)

        return R_corrupt.flatten(), R

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class Det(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = 1

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        R_corrupt = np.random.uniform(0, 1, (self.h, self.w))
        R = np.linalg.det(R_corrupt) * 10

        return R_corrupt.flatten(), np.array([R])

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class ShortestPath(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank=20, num_steps=10, vary=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.rank = rank
        self.edge_prob = 0.3
        self.vary = vary

        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w
        self.split = split
        self.num_steps = num_steps

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        # if self.split == "train":
        #     np.random.seed(index % 10000)

        if self.vary:
            rank = random.randint(2, self.rank)
        else:
            rank = self.rank

        graph = np.random.uniform(0, 1, size=[rank, rank])
        graph = graph + graph.transpose()
        np.fill_diagonal(graph, 0)  # can move to self

        graph_dist, graph_predecessors = shortest_path(csgraph=csr_matrix(
            graph), unweighted=False, directed=False, return_predecessors=True)

        return graph.flatten(), graph_dist.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class Sort(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank=20, num_steps=10):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.edge_prob = 0.3
        self.num_steps = num_steps

        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w
        self.split = split

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        x = np.random.uniform(-1, 1, self.inp_dim)
        rix = np.random.permutation(x.shape[0])
        x = x[rix]
        x_sort = np.sort(x)

        return x, x_sort

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class Eigen(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank=20, num_steps=10):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.edge_prob = 0.3
        self.num_steps = num_steps

        self.inp_dim = self.h * self.w
        self.out_dim = self.h
        self.split = split

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        x = np.random.uniform(-1, 1, (self.h, self.w))
        x = (x + x.transpose()) / 2
        x = np.matmul(x, x.transpose())
        val, eig = np.linalg.eig(x)
        rix = np.argsort(np.abs(val))[::-1]

        val = val[rix]
        eig = eig[:, rix]

        vecs = []
        vec = np.random.uniform(-1, 1, (self.h, 1))
        vecs.append(vec)
        eig = eig[:, 0]
        sign = np.sign(eig[0])
        eig = eig * sign

        return x.flatten(), eig

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class QR(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank=20, num_steps=10):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.edge_prob = 0.3

        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w * 2
        self.split = split
        self.num_steps = num_steps

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        x = np.random.uniform(-1, 1, (self.h, self.w))
        q, r = np.linalg.qr(x)
        flat = np.concatenate([q.flatten(), r.flatten()])

        flat_list = np.tile(flat[None, :], (self.num_steps, 1))
        flat_start = np.random.uniform(-1, 1, flat.shape)
        flat_list = np.concatenate([flat_start[None, :], flat_list], axis=0)

        return x.flatten(), flat  # , flat_list

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class Parity(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank=20, num_steps=10):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.edge_prob = 0.3

        self.inp_dim = rank
        self.out_dim = 1
        self.split = split
        self.num_steps = num_steps

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        val = np.random.uniform(0, 1, size=[self.inp_dim])
        mask = (val > 0.5).astype(np.float32).flatten()
        parity = mask.sum() % 2
        parity = np.array([parity])

        rand_val = np.random.uniform(0, 1) > 0.5
        parity_noise = np.array([rand_val])

        return mask, parity

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


if __name__ == "__main__":
    A = np.random.uniform(-1, 1, (5, 5))
    A = 0.5 * (A + A.transpose())
    A = np.matmul(A, A.transpose())
    x = np.random.uniform(-1, 1, (5, 1))
    b = np.random.uniform(-1, 1, (5, 1))
    sol = conjgrad(A, b, x)
    import pdb
    pdb.set_trace()
    print(A)
    print(sol)
    print(b)
