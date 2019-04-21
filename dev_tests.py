from memn2n.nn import MatVecProd
from memn2n_pytorch.nn import MatVecProdPytorch
import numpy as np
import copy

import torch


# compare MetVecPrd and MatVecProdPytorch with and without transpose option set
"""
looks like the input for MatVecProd is a list of numpy arrays
with fist being matrix (50, 50, 32) and second being a vector (50, 32)
"""

USE_CUDA = False

testcount = 2
TEST0 = True
TEST1 = True

tests = [None for i in range(testcount)]

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

matrix_batch_dim = (50, 50, 32)
vect_batch_dim = (50, 32)

# test numpy -> FloatTensor ->

if TEST0:
    tests[0] = True
    a1 = np.random.rand(2).astype('f')
    b = torch.from_numpy(a1).type(FloatTensor)
    try:
        a2 = b.data.numpy()
        assert np.array_equal(a1, a2) is True
    except AssertionError:
        tests[0] = False

if TEST1:
    tests[1] = True
    for i in range(10):
        M = np.random.rand(*matrix_batch_dim)
        V = np.random.rand(*vect_batch_dim)

        input_data = [M, V]
        input_data_torch = [torch.from_numpy(M).type(FloatTensor), torch.from_numpy(V).type(FloatTensor)]
        mvp = MatVecProd(True)
        mvp_pt = MatVecProdPytorch(True)
        result_1 = mvp.fprop(input_data)
        result_2 = mvp_pt.forward(input_data_torch)
        try:
            result_2_np = result_2.data.numpy()
            assert np.allclose(result_1, result_2_np)
        except AssertionError:
            tests[1] = False

for i in range(len(tests)):
    if tests[i] is True:
        stat_str = 'PASS'
    elif tests[i] is False:
        stat_str = 'FAIL'
    else:
        stat_str = 'NO RUN'
    print('test{}: {}'.format(i, stat_str))
