import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import re
import nltk
import codecs
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
"""
# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#
"""
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
from gensim import corpora, models, similarities
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import defaultdict

#質問を形態素解析して単語拡張を行う
#この結果をリストとして返す、リストの一つの要素内の一番初めの単語が質問の単語、それ以降は類似度順
def words_extension(Question_words):
    # word2vecの読み込み
    w2v_model = word2vec.Word2Vec.load("../../src/gui/word2vec/word2vec.gensim.model")
    # 質問の単語リスト
    Q_words = [Question_words]
    # Q_wordsの拡張結果
    Q_words_extension_list = []
    # 質問の単語リストを形態素解析
    Q_words_morpho_result = []
    # MeCab
    m = MeCab.Tagger ("-Ochasen")
    # 文単位で形態素解析し、特定の品詞だけ抽出し、基本形を文ごとのリストにする
    Q_words_morpho = [ \
        [v.split()[2] for v in m.parse(sentense).splitlines() \
           if (len(v.split())>=3 and v.split()[3][:2] in ['名詞', '形容', '動詞' , '形容動詞'])] \
        for sentense in Q_words]

    for Q_word in Q_words_morpho[0]:
        # Q_wordsの拡張結果の一時置
        Q_words_extension_list_temp = []
        Q_words_extension_list_temp.append(Q_word)
        #print(Q_word)
        Q_words_morpho_result.append(Q_word)
        #print(w2v_model.most_similar(positive=[Q_word]))
        extend_word_list = []
        w2v_model_result = w2v_model.most_similar(positive=[Q_word])[:5]
        for extend_word in w2v_model_result:
            #print(extend_word[0])
            extend_word_list.append(extend_word[0])
            Q_words_extension_list_temp.append(extend_word[0])
        Q_words_extension_list.append(Q_words_extension_list_temp)
    return Q_words_extension_list

#入力した単語に対して形態素解析を行い、結果をリストで返す
def words_morpho(Question_words):
    # word2vecの読み込み
    #w2v_model = gensim.models.Word2Vec.load("../../data/model/word2vec.gensim.model")
    # 質問の単語リスト
    Q_words =[Question_words]
    # Q_wordsの拡張結果
    Q_words_extension_list = []
    # 質問の単語リストを形態素解析
    Q_words_morpho_result = []
    # MeCab
    m = MeCab.Tagger ("-Ochasen")
# 文単位で形態素解析し、特定の品詞だけ抽出し、基本形を文ごとのリストにする
    Q_words_morpho = [ \
        [v.split()[2] for v in m.parse(sentense).splitlines() \
           if (len(v.split())>=3 and v.split()[3][:2] in ['名詞', '形容', '動詞' , '形容動詞'])] \
        for sentense in Q_words]
    return Q_words_morpho[0]
