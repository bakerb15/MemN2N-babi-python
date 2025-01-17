import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy


from memn2n_pytorch.nn import ElemMultPytorch, SumPytorch, MatVecProdPytorch, Identity, FloatToInt, Parallel, LookupTable


class Memory(nn.Module):
    """
    Memory:
        Query module  = Parallel(LookupTable + Identity) + MatVecProd with transpose + Softmax
        Output module = Parallel(LookupTable + Identity) + MatVecProd
    """

    def __init__(self, train_config):
        super(Memory, self).__init__()

        self.sz = train_config["sz"]
        self.voc_sz = train_config["voc_sz"]
        self.in_dim = train_config["in_dim"]
        self.out_dim = train_config["out_dim"]
        self.ltype = train_config["LongTensor"]

        # TODO: Mark self.nil_word and self.data as None since they will be overriden eventually
        # In build.model.py, memory[i].nil_word = dictionary['nil']"
        self.nil_word = train_config["voc_sz"]
        self.config = train_config
        self.data = nn.Parameter(train_config["FloatTensor"](np.zeros((self.sz, train_config["bsz"]), np.float32)))

        self.emb_query = None
        self.emb_out = None
        self.mod_query = None
        self.mod_out = None
        self.probs = None
        self.output = None

        self.init_query_module()
        self.init_output_module()


    def init_query_module(self):
        """
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        p = Parallel()
        p.add(self.emb_query)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())
        """
        pass

    def init_output_module(self):
        """
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        p = Parallel()
        p.add(self.emb_out)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))
        """
        pass

    def init_weights(self):
        """
        The original model used 0.1 * np.random.standard_normal(sz) to initialize Weight objects
        :return: None
        """
        standdev = 0.089
        for mdl in self.mod_query.modules():
            if isinstance(mdl, nn.Embedding):
                torch.nn.init.normal_(mdl.weight, std=standdev)
                #torch.nn.init.xavier_uniform(mdl.weight)
            elif isinstance(mdl, nn.Linear):
                torch.nn.init.normal_(mdl.weight, std=standdev)
            elif isinstance(mdl, nn.Module) and not isinstance(mdl, ElemMultPytorch) and not isinstance(mdl, LookupTable):
                if hasattr(mdl, 'weight'):
                    torch.nn.init.normal_(mdl.weight, std=standdev)


        for mdl in self.mod_out.modules():
            if isinstance(mdl, nn.Embedding):
                torch.nn.init.normal_(mdl.weight, std=standdev)
                #torch.nn.init.xavier_uniform(mdl.weight)
                # with torch.no_grad():
                #    mdl.weight = nn.Parameter(torch.from_numpy(0.1 * np.random.standard_normal(mdl.weight.shape)).type(torch.FloatTensor))
            elif isinstance(mdl, nn.Linear):
                torch.nn.init.normal_(mdl.weight, std=standdev)
                # with torch.no_grad():
                # dl.weight = nn.Parameter(torch.from_numpy(0.1 * np.random.standard_normal(mdl.weight.shape)).type(torch.FloatTensor))
            elif isinstance(mdl, nn.Module) and not isinstance(mdl, ElemMultPytorch) and not isinstance(mdl, LookupTable):
                if hasattr(mdl, 'weight'):
                    torch.nn.init.normal_(mdl.weight, std=standdev)

    def reset(self):
        self.data[:] = self.nil_word

    def put(self, data_row):
        self.data[1:, :] = self.data[:-1, :]  # shift rows down
        self.data[0, :] = data_row  # add the new data row on top

    def forward(self, input_data):
        self.probs = self.mod_query.forward([self.data, input_data])
        self.output = self.mod_out.forward([self.data, self.probs])
        return self.output

    """
    def fprop(self, input_data):
        self.probs = self.mod_query.fprop([self.data, input_data])
        self.output = self.mod_out.fprop([self.data, self.probs])
        return self.output

    def bprop(self, input_data, grad_output):
        g1 = self.mod_out.bprop([self.data, self.probs], grad_output)
        g2 = self.mod_query.bprop([self.data, input_data], g1[1])
        self.grad_input = g2[1]
        return self.grad_input
    """
    """
    def update(self, params):
        self.mod_out.update(params)
        self.mod_query.update(params)
        self.emb_out.weight.D[:, self.nil_word] = 0
    """

    def share(self, m):
        pass

class MemoryBoW(Memory):
    """
    MemoryBoW:
        Query module  = Parallel((LookupTable + Sum(1)) + Identity) + MatVecProd with transpose + Softmax
        Output module = Parallel((LookupTable + Sum(1)) + Identity) + MatVecProd
    """
    pass
    """
    def __init__(self, config):
        super(MemoryBoW, self).__init__(config)
        self.data = np.zeros((config["max_words"], self.sz, config["bsz"]), np.float32)

    def init_query_module(self):
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        s = Sequential()
        s.add(self.emb_query)
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())

    def init_output_module(self):
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        s = Sequential()
        s.add(self.emb_out)
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))
    """

class MemoryL(Memory):
    """
    MemoryL:
        Query module  = Parallel((LookupTable + ElemMult + Sum(1)) + Identity) + MatVecProd with transpose + Softmax
        Output module = Parallel((LookupTable + ElemMult + Sum(1)) + Identity) + MatVecProd
    """

    def __init__(self, train_config):
        super(MemoryL, self).__init__(train_config)
        #self.data = np.zeros((train_config["max_words"], self.sz, train_config["bsz"]), np.float32)
        self.data = nn.Parameter(train_config["FloatTensor"](torch.zeros((train_config["max_words"], self.sz, train_config["bsz"]), dtype=torch.float32)))


    def init_query_module(self):

        """ original
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        s = Sequential()
        s.add(self.emb_query)
        s.add(ElemMult(self.config["weight"]))
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())
        """

        #self.emb_query = nn.Embedding(self.voc_sz, self.in_dim, sparse=True)
        self.emb_query = LookupTable(self.voc_sz, self.out_dim)
        emb_query_layers = [
            FloatToInt(self.ltype),
            self.emb_query,
            ElemMultPytorch(self.config["weight"]),
            SumPytorch(dim=1)]
        # s = nn.Sequential(*emb_query_layers)

        p = Parallel(emb_query_layers, [Identity()])

        mod_query_layers = [
            p,
            MatVecProdPytorch(True),
            nn.Softmax()]
        self.mod_query = nn.Sequential(*mod_query_layers)





    def init_output_module(self):
        """
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        s = Sequential()
        s.add(self.emb_out)
        s.add(ElemMult(self.config["weight"]))
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))
        """

        #self.emb_out = nn.Embedding(self.voc_sz, self.out_dim, sparse=True)
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        emb_query_layers = [
            FloatToInt(self.ltype),
            self.emb_out,
            ElemMultPytorch(self.config["weight"]),
            SumPytorch(dim=1)]
        # s = nn.Sequential(*emb_query_layers)

        p = Parallel(emb_query_layers, [Identity()])

        mod_query_layers = [
            p,
            MatVecProdPytorch(False)]
        self.mod_out = nn.Sequential(*mod_query_layers)
