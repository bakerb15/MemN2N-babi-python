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
        super(MatVecProd, self).__init__()
        self.do_transpose = do_transpose

    def forward(self, input_data):
        M = input_data[0]
        V = input_data[1]

        if self.do_transpose:
            pass
        else:
            pass