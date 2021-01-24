import networkx as nx
# matplotlibのターミナル対応
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import re
import nltk
import codecs
"""
# xml スクレイピング
from bs4 import BeautifulSoup
import xml.etree.ElementTree as et
"""
# 形態素解析
import sys
import MeCab
import MeCab as mc
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import models
from gensim.models import word2vec
from copy import copy
import jaconv
import gensim
import warnings
warnings.filterwarnings('ignore')#警告無視

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
# 因子分析
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from DocumentFeatureSelection import interface
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize
import itertools
from collections import Counter
from itertools import combinations, dropwhile
from collections import Counter, OrderedDict
from gensim import corpora, models, similarities
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import defaultdict


"""
kyoki=pd.read_csv('df_candid_vector_vector.csv')
kyoki_list=kyoki['sentence_morpho']

kyoki_all_list=[]

for i in range(len(kyoki_list)):
    list_temp = kyoki_list[i].replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')
    #print(type(list_temp))
    kyoki_all_list.append(list_temp)

print(kyoki_all_list)

def bform2pair(bform_2l, min_cnt=10):
    # 単語ペアの出現章数をカウント

    # 全単語ペアのリスト
    pair_all = []

    for bform_l in bform_2l:
        # 章ごとに単語ペアを作成
        # combinationsを使うと順番が違うだけのペアは重複しない
        # ただし、同単語のペアは存在しえるのでsetでユニークにする
        pair_l = list(combinations(set(bform_l), 2))

        # 単語ペアの順番をソート
        for i,pair in enumerate(pair_l):
            pair_l[i] = tuple(sorted(pair))

        pair_all += pair_l

    # 単語ペアごとの出現章数
    pair_count = Counter(pair_all)

    # ペア数がmin_cnt以上に限定
    for key, count in dropwhile(lambda key_count: key_count[1] >= min_cnt, pair_count.most_common()):
        del pair_count[key]

    return pair_count

def save_dict_to_file(dic):
    f = open('kyoki_counter.txt','w')
    f.write(str(dic))
    f.close()

def load_dict_from_file():
    f = open('kyoki_counter.txt','r')
    kyoki_counter_new=f.read()
    f.close()
    return eval(kyoki_counter_new)


kyoki_counter = bform2pair(kyoki_all_list, min_cnt=10)

print(kyoki_counter)
print(type(kyoki_counter))

save_dict_to_file(kyoki_counter)

kyoki_counter_new=load_dict_from_file()


def pair2jaccard(pair_count, bform_2l, edge_th=0.2):
    # jaccard係数を計算

    # 単語ごとの出現章数
    word_count = Counter()
    for bform_l in bform_2l:
        word_count += Counter(set(bform_l))

    # 単語ペアごとのjaccard係数を計算
    jaccard_coef = []
    for pair, cnt in pair_count.items():
        k = word_count[pair[0]] + word_count[pair[1]] - cnt
        if k == 0:
            jaccard_coef.append(0)
        else:
            jaccard_coef.append(cnt / k)

    # jaccard係数がedge_th未満の単語ペアを除外
    jaccard_dict = OrderedDict()
    for (pair, cnt), coef in zip(pair_count.items(), jaccard_coef):
        if coef >= edge_th:
            jaccard_dict[pair] = coef
            print(pair, cnt, coef, word_count[pair[0]], word_count[pair[1]], sep='\t')

    return jaccard_dict

jaccard_dict= pair2jaccard(kyoki_counter_new, kyoki_all_list, edge_th=0.2)


def save_dict_toto_file(dic):
    f = open('jaccard_dict.txt','w')
    f.write(str(dic))
    f.close()

"""

def load_dict_fromfrom_file():
    f = open('jaccard_dict.txt','r')
    jaccard_dict_new=f.read()
    f.close()
    return eval(jaccard_dict_new)

def inverse_lookup(d, x):
    kk=[]
    for k,v in d.keys():
        if x == v:
            kk.append(k)
        elif x == k:
            kk.append(v)
        else:
            a=0
    return kk
x=('電気')
jaccard_dict_new=load_dict_fromfrom_file()
print(type(jaccard_dict_new))
kyoki_list=inverse_lookup(jaccard_dict_new, x)
print(kyoki_list)
