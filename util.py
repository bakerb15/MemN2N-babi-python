from __future__ import division

import sys
import time
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from memn2n.memory import MemoryL, MemoryBoW
from memn2n.nn import AddTable, CrossEntropyLoss, Duplicate, ElemMult, LinearNB
from memn2n.nn import Identity, ReLU, Sequential, LookupTable, Sum, Parallel, Softmax

from memn2n_pytorch.memory import MemoryL as ptMemoryL
from memn2n_pytorch.memory import SumPytorch
from memn2n_pytorch.nn import ElemMultPytorch
from memn2n_pytorch.nn import Duplicate as ptDuplicate
from memn2n_pytorch.nn import Identity as ptIdentity
from memn2n_pytorch.nn import AddTable as ptAddTable

class DataType(Enum):
    NUMPY = 1
    PYTORCH = 2
    PYTORCH_CUDA = 3


def parse_babi_task(data_files, dictionary, include_question, dt=DataType.NUMPY):
    """ Parse bAbI data.

    Args:
       data_files (list): a list of data file's paths.
       dictionary (dict): word's dictionary
       include_question (bool): whether count question toward input sentence.

    Returns:
        A tuple of (story, questions, qstory):
            story (3-D array)
                [position of word in sentence, sentence index, story index] = index of word in dictionary
            questions (2-D array)
                [0-9, question index], in which the first component is encoded as follows:
                    0 - story index
                    1 - index of the last sentence before the question
                    2 - index of the answer word in dictionary
                    3 to 13 - indices of supporting sentence
                    14 - line index
            qstory (2-D array) question's indices within a story
                [index of word in question, question index] = index of word in dictionary
    """
    # Try to reserve spaces beforehand (large matrices for both 1k and 10k data sets)
    # maximum number of words in sentence = 20


    story     = np.zeros((20, 500, len(data_files) * 3500), np.int16)
    questions = np.zeros((14, len(data_files) * 10000), np.int16)
    qstory    = np.zeros((20, len(data_files) * 10000), np.int16)

    if dt == DataType.PYTORCH or dt == DataType.PYTORCH_CUDA:
        USE_CUDA = dt == DataType.PYTORCH_CUDA
        FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        story = torch.from_numpy(story).type(FloatTensor)
        questions = torch.from_numpy(questions).type(FloatTensor)
        qstory = torch.from_numpy(qstory).type(FloatTensor)



    # NOTE: question's indices are not reset when going through a new story
    story_idx, question_idx, sentence_idx, max_words, max_sentences = -1, -1, -1, 0, 0

    # Mapping line number (within a story) to sentence's index (to support the flag include_question)
    mapping = None

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip().lower()
                words = line.split()

                # Story begins
                if words[0] == '1':
                    story_idx += 1
                    sentence_idx = -1
                    mapping = []

                # FIXME: This condition makes the code more fragile!
                if '?' not in line:
                    is_question = False
                    sentence_idx += 1
                else:
                    is_question = True
                    question_idx += 1
                    questions[0, question_idx] = story_idx
                    questions[1, question_idx] = sentence_idx
                    if include_question:
                        sentence_idx += 1

                mapping.append(sentence_idx)

                # Skip substory index
                for k in range(1, len(words)):
                    w = words[k]

                    if w.endswith('.') or w.endswith('?'):
                        w = w[:-1]

                    if w not in dictionary:
                        dictionary[w] = len(dictionary)

                    if max_words < k:
                        max_words = k

                    if not is_question:
                        story[k - 1, sentence_idx, story_idx] = dictionary[w]
                    else:
                        qstory[k - 1, question_idx] = dictionary[w]
                        if include_question:
                            story[k - 1, sentence_idx, story_idx] = dictionary[w]

                        # NOTE: Punctuation is already removed from w
                        if words[k].endswith('?'):
                            answer = words[k + 1]
                            if answer not in dictionary:
                                dictionary[answer] = len(dictionary)

                            questions[2, question_idx] = dictionary[answer]

                            # Indices of supporting sentences
                            for h in range(k + 2, len(words)):
                                questions[1 + h - k, question_idx] = mapping[int(words[h]) - 1]

                            questions[-1, question_idx] = line_idx
                            break

                if max_sentences < sentence_idx + 1:
                    max_sentences = sentence_idx + 1

    story     = story[:max_words, :max_sentences, :(story_idx + 1)]
    questions = questions[:, :(question_idx + 1)]
    qstory    = qstory[:max_words, :(question_idx + 1)]

    return story, questions, qstory


def build_model(general_config):
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

    if not use_bow:
        train_config["weight"] = np.ones((in_dim, max_words), np.float32)
        for i in range(in_dim):
            for j in range(max_words):
                train_config["weight"][i][j] = (i + 1 - (in_dim + 1) / 2) * \
                                               (j + 1 - (max_words + 1) / 2)
        train_config["weight"] = \
            1 + 4 * train_config["weight"] / (in_dim * max_words)

    memory = {}
    model = Sequential()
    model.add(LookupTable(voc_sz, in_dim))
    if not use_bow:
        if enable_time:
            model.add(ElemMult(train_config["weight"][:, :-1]))
        else:
            model.add(ElemMult(train_config["weight"]))

    model.add(Sum(dim=1))

    proj = {}
    for i in range(nhops):
        if use_bow:
            memory[i] = MemoryBoW(train_config)
        else:
            memory[i] = MemoryL(train_config)

        # Override nil_word which is initialized in "self.nil_word = train_config["voc_sz"]"
        memory[i].nil_word = dictionary['nil']
        model.add(Duplicate())
        p = Parallel()
        p.add(memory[i])

        if add_proj:
            proj[i] = LinearNB(in_dim, in_dim)
            p.add(proj[i])
        else:
            p.add(Identity())

        model.add(p)
        model.add(AddTable())
        if add_nonlin:
            model.add(ReLU())

    model.add(LinearNB(out_dim, voc_sz, True))
    model.add(Softmax())

    # Share weights
    if share_type == 1:
        # Type 1: adjacent weight tying
        memory[0].emb_query.share(model.modules[0])
        for i in range(1, nhops):
            memory[i].emb_query.share(memory[i - 1].emb_out)

        model.modules[-2].share(memory[len(memory) - 1].emb_out)

    elif share_type == 2:
        # Type 2: layer-wise weight tying
        for i in range(1, nhops):
            memory[i].emb_query.share(memory[0].emb_query)
            memory[i].emb_out.share(memory[0].emb_out)

    if add_proj:
        for i in range(1, nhops):
            proj[i].share(proj[0])

    # Cost
    loss = CrossEntropyLoss()
    loss.size_average = False
    loss.do_softmax_bprop = True
    model.modules[-1].skip_bprop = True

    return memory, model, loss


def build_model_pytorch(general_config):
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

    if not use_bow:
        train_config["weight"] = np.ones((in_dim, max_words), np.float32)
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

    # pytorch #1
    memory = {}
    mlayers = [nn.Embedding(voc_sz, in_dim)]
    if not use_bow:
        if enable_time:
            mlayers.append(ElemMultPytorch(train_config["weight"][:, :-1]))
        else:
            mlayers.append(ElemMultPytorch(train_config["weight"]))
    mlayers.append(SumPytorch(dim=1))

    # end pytorch #1


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
        #p = Parallel() # disregard for now
        mlayers.append(memory[i])   #p.add(memory[i])

        if add_proj:
            proj[i] = nn.Linear(in_dim, in_dim, bias=False) #LinearNB(in_dim, in_dim)
            mlayers.append(proj[i])
        else:
            mlayers.append(ptIdentity)

        #model.add(p)
        mlayers.append(ptAddTable()) #model.add(AddTable())
        if add_nonlin:
            mlayers.append(nn.ReLU()) #model.add(ReLU())

    mlayers.append(nn.Linear(out_dim, voc_sz, bias=False )) #model.add(LinearNB(out_dim, voc_sz, True))

    # no need to apply softmax because pytorch CrossEntropyLoss does so automatically
    #mlayers.append(nn.Softmax()) # model.add(Softmax())

    model = nn.Sequential(*mlayers)



    # Share weights
    if share_type == 1:
        # Type 1: adjacent weight tying
        memory[0].emb_query.weight = model.modules()[0].weight # memory[0].emb_query.share(model.modules[0])
        for i in range(1, nhops):
            memory[i].emb_query.weight = memory[i - 1].emb_out.weight # memory[i].emb_query.share(memory[i - 1].emb_out)

        model.modules()[-2].weight = memory[len(memory) - 1].emb_out.weight # model.modules[-2].share(memory[len(memory) - 1].emb_out)

    elif share_type == 2:
        # Type 2: layer-wise weight tying
        if add_nonlin:
            mlayers.append(nn.ReLU()) #model.add(ReLU())

    mlayers.append(nn.Linear(out_dim, voc_sz, bias=False )) #model.add(LinearNB(out_d
    for i in range(1, nhops):
        memory[i].emb_query.weight = memory[0].emb_query.weight  # memory[i].emb_query.share(memory[0].emb_query)
        memory[i].emb_out.weight = memory[0].emb_out.weight  # memory[i].emb_out.share(memory[0].emb_out)

    if add_proj:
        for i in range(1, nhops):
            proj[i].weight = proj[0].weight # proj[i].share(proj[0])

    # Cost
    # loss.size_average = False
    loss = nn.CrossEntropyLoss(size_average=False) # loss = CrossEntropyLoss()

    #loss.do_softmax_bprop = True
    #model.modules[-1].skip_bprop = True


    return memory, model, loss

class Progress(object):
    """
    Progress bar
    """

    def __init__(self, iterable, bar_length=50):
        self.iterable = iterable
        self.bar_length = bar_length
        self.total_length = len(iterable)
        self.start_time = time.time()
        self.count = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.count += 1
            percent = self.count / self.total_length
            print_length = int(percent * self.bar_length)
            progress = "=" * print_length + " " * (self.bar_length - print_length)
            elapsed_time = time.time() - self.start_time
            print_msg = "\r|%s| %.0f%% %.1fs" % (progress, percent * 100, elapsed_time)
            sys.stdout.write(print_msg)
            if self.count == self.total_length:
                sys.stdout.write("\r" + " " * len(print_msg) + "\r")
            sys.stdout.flush()
