import torch
from torch import nn
import numpy as np

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

        if len(input_data.shape) == 2:
            output = input_data.view(50, 11,) * self.weight
        elif len(input_data.shape) == 3:
            #output = input_data.view(50, 11, -1) * self.weight[:, :, None]  #
            dim0 =  input_data.shape[2]
            dim1 =  input_data.shape[0]
            dim2 = -1
            output = input_data.view(dim0, dim1, dim2) * self.weight[:, :, None]  # broadcasting
        elif len(input_data.shape) == 4:
            #output = input_data.view(50, 11, -1, -1) * self.weight[:, :, None, None]  # broadcasting
            dim0 = input_data.shape[1]
            dim1 = input_data.shape[0]
            dim2 = input_data.shape[3]
            dim3 = -1
            output = input_data.view(dim0, dim1, dim2, dim3) * self.weight[:, :, None, None]  # broadcasting
        else:
            raise Exception("input_data has large dimension = %d" % input_data.ndim)
        return output

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
        #return input_data.sum(0)
        output = input_data[0]
        for elem in input_data[1:]:
            # Expand to the same ndim as self.output
            if len(elem.shape) == len(output.shape) - 1:
                elem.unsqueeze_(-1)
                elem = elem.expand(output.shape[0], output.shape[1], output.shape[2])
            output += elem
        return output


class Parallel(nn.Module):
    """
    Computes forward and backward propagations for all modules at once.
    """
    def __init__(self):
        super(Parallel, self).__init__()
        self.mds = []

    def add(self, pytorch_module):
        self.mds.append(pytorch_module)

    def forward(self, input_data):
        output = [md.forward(input_elem)
                       for md, input_elem in zip(self.mds, input_data)]
        return output




class FloatToInt(nn.Module):

    def __init__(self, ltype):
        super(FloatToInt, self).__init__()
        self.LTYPE = ltype
    def forward(self, input_data):
        if type(input_data) is list:
            return [x.long() for x in input_data]
        else:
            return input_data.long()


class LinearNB(nn.Module):
    """
    Linear layer with no bias
    """
    def __init__(self, in_dim, out_dim, do_transpose=False):
        super(LinearNB, self).__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.do_transpose = do_transpose

        if do_transpose:
            lin_mod = nn.Linear(in_dim, out_dim, bias=False)
        else:
            lin_mod = nn.Linear(out_dim, in_dim, bias=False)

        self.m = lin_mod

    def forward(self, input_data):
        high_dimension_input = len(input_data.shape) > 2

        if high_dimension_input:
            input_data = input_data.reshape(input_data.shape[0], -1)

        if self.do_transpose:
            output = torch.matmul(self.m.weight, input_data)
        else:
            output = torch.matmul(self.m.weight.t(), input_data)

        if high_dimension_input:
            output = output.view(self.output.shape[0], -1)

        return output
