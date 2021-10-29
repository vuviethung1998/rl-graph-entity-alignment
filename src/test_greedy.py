import time
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import string
from tqdm import tqdm
import os
from framework.greedy.kgs import *
from framework.greedy.evaluation import *
from framework.utils import *

def test(save=True, simi_matrix=None):
    top_k = [1, 5, 10, 50]
    test_threads_num = 16
    eval_metric = 'cosine'
    eval_norm = True
    csls = 10
    
    embeds1, embeds2, unique1, unique2 = build_embeddings()
    alignment_rest_12, hits1_12, mr_12, mrr_12 = test_(embeds1, embeds2, unique1, unique2,None, top_k, test_threads_num,
            metric=eval_metric, normalize=eval_norm, csls_k=csls, accurate=True)
    return alignment_rest_12, hits1_12, mr_12, mrr_12

if __name__=="__main__":
    alignment_rest_12, hits1_12, mr_12, mrr_12 = test()
    # print(alignment_rest_12)
