import torch
from torch import nn

""" Online example taken from https://stackoverflow.com/questions/51980654/pytorch-element-wise-filter-layer

import torch
from torch import nn

class TrainableEltwiseLayer(nn.Module)
  def __init__(self, n, h, w):
    super(TrainableEltwiseLayer, self).__init__()
    self.weights = nn.Parameter(torch.tensor(1, n, h, w))  # define the trainable parameter

  def forward(self, x):
    # assuming x is of size b-1-h-w
    return x * self.weights  # element-wise multiplication

"""


class ElemMultPytorch(nn.Module):
    def __init__(self, weight):
        super(ElemMultPytorch, self).__init__()
        # self.weight = nn.Parameter(torch.tensor(1, n, h, w))  # define the trainable parameter
        self.weight = nn.Parameter(weight)

    def forward(self, input_data):
        # assuming x is of size b-1-h-w
        return input_data * self.weight  # element-wise multiplication


class SumPytorch(nn.Module):
    def __init__(self, dim):
        super(SumPytorch, self).__init__()
        self.dim = dim

    def forward(self, input_data):
        return torch.sum(input_data, dim=self.dim)


class MatVecProdPytorch(nn.Module):
    """
        Product of matrix and vector in batch, where
            matrix's shape is [:, :, batch] and vectors is [:, batch]
        Result is a vector of size [:, batch]
        """

    def __init__(self, do_transpose):
        super(MatVecProdPytorch, self).__init__()
        self.do_transpose = do_transpose

    def forward(self, input_data):
        M = input_data[0]
        V = input_data[1]

        batch_size = M.shape[2]

        if self.do_transpose:
            output = torch.zeros((M.shape[1], batch_size)).type(M.dtype)
            for i in range(batch_size):
                output[:, i] = torch.matmul(M[:, :, i].t(), V[:, i])
        else:
            output = torch.zeros((M.shape[0], batch_size)).type(M.dtype)
            for i in range(batch_size):
                output[:, i] = torch.matmul(M[:, :, i], V[:, i])

        return output


class Duplicate(nn.Module):

    def __init__(self):
        super(Duplicate, self).__init__()

    def forward(self, input_data):
        return [input_data, input_data]


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input_data):
        return input_data

class AddTable(nn.Module):
    """
    Module for sum operator which sums up all elements in input data
    """
    def __init__(self):
        super(AddTable, self).__init__()

    def forward(self, input_data):
        return input_data.sum(0)