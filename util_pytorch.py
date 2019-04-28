from __future__ import division

import sys
import time
from enum import Enum

import numpy as np
import torch
import torch.nn as nn



from memn2n_pytorch.memory import MemoryL as ptMemoryL
from memn2n_pytorch.memory import SumPytorch
from memn2n_pytorch.nn import ElemMultPytorch
from memn2n_pytorch.nn import Duplicate as ptDuplicate
from memn2n_pytorch.nn import Identity as ptIdentity
from memn2n_pytorch.nn import AddTable as ptAddTable
from memn2n_pytorch.nn import FloatToInt, Parallel, LinearNB


def build_model_pytorch(general_config, USE_CUDA=False):
    """
    Build model

    NOTE: (for default config)
    1) Model's architecture (embedding B)
        LookupTable -> ElemMult -> Sum -> [ Duplicate -> { Parallel -> Memory -> Identity } -> AddTable ] -> LinearNB -> Softmax

    2) Memory's architecture
        a) Query module (embedding A)
            Parallel -> { LookupTable + ElemMult + Sum } -> Identity -> MatVecProd -> Softmax

        b) Output module (embedding C)
            Parallel -> { LookupTable + ElemMult + Sum } -> Identity -> MatVecProd
    """
    train_config = general_config.train_config
    dictionary   = general_config.dictionary
    use_bow      = general_config.use_bow
    nhops        = general_config.nhops
    add_proj     = general_config.add_proj
    share_type   = general_config.share_type
    enable_time  = general_config.enable_time
    add_nonlin   = general_config.add_nonlin

    in_dim    = train_config["in_dim"]
    out_dim   = train_config["out_dim"]
    max_words = train_config["max_words"]
    voc_sz    = train_config["voc_sz"]

    FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    if not use_bow:
        # train_config["weight"] = np.ones((in_dim, max_words), np.float32)
        train_config["weight"] = FloatTensor(torch.ones((in_dim, max_words), dtype=torch.float32))
        for i in range(in_dim):
            for j in range(max_words):
                train_config["weight"][i][j] = (i + 1 - (in_dim + 1) / 2) * \
                                               (j + 1 - (max_words + 1) / 2)
        train_config["weight"] = \
            1 + 4 * train_config["weight"] / (in_dim * max_words)

    """ original #1
    memory = {}
    model = Sequential()
    model.add(LookupTable(voc_sz, in_dim))
    if not use_bow:
        if enable_time:
            model.add(ElemMult(train_config["weight"][:, :-1]))
        else:
            model.add(ElemMult(train_config["weight"]))

    model.add(Sum(dim=1))
    """

    memory = {}
    modle_emb = nn.Embedding(voc_sz, in_dim)
    mlayers = [FloatToInt(train_config["LongTensor"]), modle_emb]
    if not use_bow:
        if enable_time:
            mlayers.append(ElemMultPytorch(train_config["weight"][:, :-1]))
        else:
            mlayers.append(ElemMultPytorch(train_config["weight"]))
    mlayers.append(SumPytorch(dim=1))


    proj = {}
    for i in range(nhops):
        if use_bow:
            print('BOW not implemented')
            assert 1 == 2
        else:
            memory[i] = ptMemoryL(train_config)

        # Override nil_word which is initialized in "self.nil_word = train_config["voc_sz"]"
        memory[i].nil_word = dictionary['nil']
        mlayers.append(ptDuplicate())# model.add(Duplicate())

        leftlayer = [memory[i]] # p.add(memory[i])
        rightlayer = []
        if add_proj:
            #proj[i] = nn.Linear(in_dim, in_dim, bias=False) #LinearNB(in_dim, in_dim)
            proj[i] = LinearNB(in_dim, in_dim)
            rightlayer.append(proj[i])  # p.add(proj[i])
        else:
            rightlayer.append(ptIdentity())  # p.add(ptIdentity())

        p = Parallel(leftlayer, rightlayer)
        mlayers.append(p)
        mlayers.append(ptAddTable()) #model.add(AddTable())
        if add_nonlin:
            mlayers.append(nn.ReLU()) #model.add(ReLU())

    #model_last_linear = nn.Linear(out_dim, voc_sz, bias=False )
    model_last_linear = LinearNB(out_dim, voc_sz, True)
    mlayers.append(model_last_linear) #model.add(LinearNB(out_dim, voc_sz, True))

    # no need to apply softmax because pytorch CrossEntropyLoss does so automatically
    #mlayers.append(nn.Softmax()) # model.add(Softmax())

    model = nn.Sequential(*mlayers)



    # Share weights
    if share_type == 1:
        # Type 1: adjacent weight tying
        memory[0].emb_query.weight = modle_emb.weight # memory[0].emb_query.share(model.modules[0])
        for i in range(1, nhops):
            memory[i].emb_query.weight = memory[i - 1].emb_out.weight  # memory[i].emb_query.share(memory[i - 1].emb_out)
        '''
        the original model grabs the second to last module where as pytorch
            last module is Linear with no bias because Softmax included in Crossentropy loss
        '''
        model_last_linear.weight = memory[len(memory) - 1].emb_out.weight # model.modules[-2].share(memory[len(memory) - 1].emb_out)

    elif share_type == 2:
        pass
        '''
        for i in range(1, nhops):
            memory[i].emb_query.weight = memory[0].emb_query.weight  # memory[i].emb_query.share(memory[0].emb_query)
            memory[i].emb_out.weight = memory[0].emb_out.weight  # memory[i].emb_out.share(memory[0].emb_out)
        '''

    if add_proj:
        for i in range(1, nhops):
            proj[i].weight = proj[0].weight # proj[i].share(proj[0])

    # Cost
    # loss.size_average = False
    loss = nn.CrossEntropyLoss(size_average=False) # loss = CrossEntropyLoss()

    #loss.do_softmax_bprop = True
    #model.modules[-1].skip_bprop = True


    return memory, model, loss