import time
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import string
from tqdm import tqdm
import os
from framework.modules.kgs import *
from framework.modules.evaluation import *
from framework.utils import *

def test(save=True, simi_matrix=None):
    top_k = [1, 5, 10, 50]
    test_threads_num = 16
    eval_metric = 'cosine'
    eval_norm = True
    csls = 10

    embeds1, embeds2, unique1, unique2 = build_embeddings()

    alignment_rest_12, hits1_12, mr_12, mrr_12 = test_(embeds1, embeds2, None, top_k, test_threads_num,
            metric=eval_metric, normalize=eval_norm, csls_k=csls, accurate=True)
    lst_alignment_idx = list(alignment_rest_12)

    alignments = [(unique1[align_idx[0]], unique2[align_idx[1]]) for align_idx in  lst_alignment_idx ]
    
    final_res = {}

    len_align = len(alignments)
    while (len_align > 0):
        tmp_dct = alignments[0]
        tmp_key, tmp_val = tmp_dct[0], tmp_dct[1]
        if tmp_key not in final_res.keys():
            final_res.update({tmp_key: [tmp_val]})
        else:
            final_res[tmp_key].append(tmp_val)
        alignments.pop(0)
        len_align -= 1
    
    return final_res, hits1_12, mr_12, mrr_12

if __name__=="__main__":
    alignment_rest_12, hits1_12, mr_12, mrr_12 = test()
    print(alignment_rest_12)
