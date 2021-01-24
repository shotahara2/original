#　必要なライブラリを読み込む
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import re
import nltk
import codecs
import time
# xml スクレイピング
from bs4 import BeautifulSoup
import xml.etree.ElementTree as et
# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from copy import copy
import jaconv
import warnings
warnings.filterwarnings('ignore')#警告無視
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
# 因子分析
#from sklearn.decomposition import FactorAnalysis
#from factor_analyzer import FactorAnalyzer
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
from wordcloud import WordCloud
import pathlib,sys

# base.pyのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
# 自作
from utils.words_extension import words_morpho

"""
# df_targetを読み込むと'sentence_morpho'のカラムのリストがstrのため、
# これをリストに変換する
# 変換後カラム：text_morpho_list
def morpho_str2list(df_target):
    start = time.time()
    text_list = []
    for num_df in range(len(df_target['sentence_morpho'])):
        text_list_temp = []
        # 対象の文章
        target_text = df_target['sentence_morpho'][num_df]
        target_text = target_text[1:len(target_text)-1]
        target_text=target_text.split(',')
        for num in range(len(target_text)):
            text_list_temp.append(target_text[num].strip("'").strip(" '"))

        text_list.append(text_list_temp)
    # リスト化して,df_targetmに追加
    df_target['text_morpho_list'] = text_list
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    #df_target を返す
    return df_target

# words_extension_listがsentence内に存在するか否かをリストで返す
# target_text_list = df_target.text_morpho_list[0]
# words_extension_list = words_extension('溶接事業者検査の対象範囲')
def exist_ex_words(target_text_list, words_extension_list):

    # 出現回数
    freq_list_ex_words = []

    # words_extension_listの長さ
    for num_ex in range(len(words_extension_list)):
        count = 0
        # 一つのwords_extension_listの中の単語
        for search_word in words_extension_list[num_ex]:
            # target_text_list内に何回でてくるのか
            for target_word in target_text_list:
                #print(target_word)
                if target_word == search_word:
                    count += 1
        freq_list_ex_words.append(count)

    # 出現したか否か
    exist_list_ex_words = []

    for num_freq in freq_list_ex_words:
        if num_freq>= 1:
            exist_list_ex_words.append(1)
        else:
            exist_list_ex_words.append(0)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return exist_list_ex_words

# exist_listに１があるかないか
# ある：１、ない：０
# exist_list = exist_ex_words(target_text_list=df_target.text_morpho_list[0],words_extension_list=words_extension_list)
def exist_list_zeroone(exist_list):
    count = 0
    for num in exist_list:
        count += num

    if num >= 1:
        return 1
    else:
        return 0

# df_targetを読み込むと'sentence_morpho'のカラムのリストがstrのため、
# これをリストに変換する
# 変換後カラム：text_morpho_list
def id_str2list(df_target, target_column_name):
    text_list = []
    for num_df in range(len(df_target[target_column_name])):
        text_list_temp = []
        # 対象の文章
        target_text = df_target[target_column_name][num_df]
        target_text = target_text[1:len(target_text)-1]
        target_text=target_text.split(',')
        for num in range(len(target_text)):
            text_list_temp.append(target_text[num])#.strip("'").strip(" '"))

        text_list.append(text_list_temp)
    # リスト化して,df_targetmに追加
    df_target[target_column_name] = text_list

    #df_target を返す
    return df_target

"""

#コサイン類似度を調べる関数
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


#データフレームを作る関数１１１１１１
#file_path_process4はcsv形式のファイル
#file_path_search_tempは書き込みデータのcsvファイルパス

def make_df_all(file_path_process4, file_path_search_temp):
    start = time.time()
    count = 0

    df_all = pd.DataFrame()
    for num_file in range(len(file_path_process4)):
        # 対象データの読み込み
        df_target = pd.read_csv(file_path_search_temp[num_file])
        #対象の行を削除
        df_target = df_target.drop('Unnamed: 0', axis=1)
        #df_target = df_target.drop(df_target.columns[[1]], axis=1)
        #インデックス変更して並び替え
        df_target.reset_index(drop=True, inplace=True)
        #データフレームを連結する
        df_all = pd.concat([df_all, df_target],axis=0)

        count += len(df_target)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return df_all


# 候補のidの根拠のidのリストを作る２２２２２２２２
def make_candid_reason_list(df_all, integ_cluster_doc_id):
    start = time.time()
    candid_reason_list_temp = []
    #reason_all_list = df_all[df_all.id.isin(integ_cluster_doc_id)]['根拠_id_all'].tolist()
    reason_all_list = df_all[df_all.id.isin(integ_cluster_doc_id)]['根拠_id_all'].tolist()

    for str_list in reason_all_list:
        #置き換え
        reason_list_temp = str_list.replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')

        for reason_id in reason_list_temp:
            if len(reason_id) != 0 and reason_id != '0':
                try:
                    candid_reason_list_temp.append(int(reason_id))
                except Exception as e:
                    e_temp=0


    candid_reason_list = list(set(candid_reason_list_temp))
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return candid_reason_list


    # 候補のidの所属のidのリストを作る３３３３３３３３３３
def make_candid_belong_list(df_all, integ_cluster_doc_id):
    start = time.time()
    candid_belong_list_temp = []
    belong_all_list = df_all[df_all.id.isin(integ_cluster_doc_id)]['所属'].tolist()

    #for belong_id in belong_all_list:
        #print(str(belong_id) == 'nan')
        #print(belong_id)

        #if str(belong_id) != 'nan' and belong_id!= 0:
            #print(int(belong_id))
            #candid_belong_list_temp.append(int(belong_id))

    candid_belong_list_temp=[belong_id for belong_id in belong_all_list if str(belong_id)!= 'nan' and belong_id!= 0]
    candid_belong_list = list(set(candid_belong_list_temp))

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return candid_belong_list

#=====================================================================
# ベクトル化と正規化(使っていない)
#=====================================================================

# 入力情報の分かち書き６６６６６６６
#デンドログラムを作る関数
def make_dendro(df_candid_vector, remove_words_list, stop_word_list_wordcloud,dendro_cluster_num,not_search_words, input_words, dict_voca_num_candid):
    start = time.time()
    input_words_morpho=words_morpho(input_words)
    not_search_words_morpho=words_morpho(not_search_words)

    # キーワードベクトルの初期化
    keywords_vector =[0.0 for i in range(len(dict_voca_num_candid))]

    # キーワードベクトルの生成
    input_word_vector_num_list = []
    for input_word in input_words_morpho:
        try:
            input_word_vector_num_list.append(dict_voca_num_candid[input_word])
        except:
            a=0

    not_search_word_vector_num_list = []
    for not_search_word in not_search_words_morpho:
        try:
            not_search_word_vector_num_list.append(dict_voca_num_candid[not_search_word])
        except:
            a=0



#不必要なキーワード
#数値をいじるべき
    for not_search_index in not_search_word_vector_num_list:
        keywords_vector[not_search_index]=-1.0

#必要なキーワード
    for input_word_index in input_word_vector_num_list:
        keywords_vector[input_word_index]=1.0




    list_freq = []

    # デンドログラムの対象データのデータフレームを作成する
    df_dendro_target = pd.DataFrame(df_candid_vector.vector_add_norm.tolist())

    # デンドログラムを作成する
    Z = linkage(df_dendro_target, method='ward', metric='euclidean')
    """
    fig = plt.figure(figsize=(8, 15), facecolor="w")
    ax = fig.add_subplot(3, 1, 1, title="樹形図: 全体")
    dendrogram(Z)
    plt.show()
    """

    #######################################################################
    # クラスタを作成する
    result = fcluster(Z, t=dendro_cluster_num, criterion='maxclust')
        #######################################################################

    # デンドログラムによってクラスタ数に分割された結果を辞書型で保存
    d = defaultdict(list)
    for i, r in enumerate(result):
        d[r].append(i)

    #######################################################################
    # デンドログラムより得られた結果をデータフレームに追加する
    # デンドログラムによるクラスタ数
    dendro_cluster_num = dendro_cluster_num
    df_candid_vector['result_dendro'] = list(fcluster(Z, t=dendro_cluster_num, criterion='maxclust'))
    #######################################################################

    # dfのサイズをみる
    size_df_list = []
    for num_size in range(dendro_cluster_num):
        size_temp = len(df_candid_vector[df_candid_vector.result_dendro==(num_size+1)])
        size_df_list.append(size_temp)

    #size_temp = len(df_candid_vector[df_candid_vector.result_dendro==(num_size+1)])
    #size_df_list=[size_temp for num_size in range(dendro_cluster_num)]

    #print('dfのサイズをみる')
    #print(size_df_list)
    """
    size_df_list_norm = []
    max_num = max(size_df_list)
    for i in range(len(size_df_list)):
        size_df_list_norm.append((size_df_list[i]/max_num)+0.5)
    """
    #max_num = max(size_df_list)
    size_df_list_norm=[(size_df_list[i]/max(size_df_list))+0.5 for i in range(len(size_df_list))]

    #print('正規化')
    #print(size_df_list_norm)

    texts = []
    texts_wordcloud = ''

    # 入力キーワードと一つのワードクラウドの類似度を格納するリスト
    keywords_sim_list = []

    for num_d in range(len(d)):
        # 入力キーワードと一つのワードクラウドの類似度を計算する
        total_count = 0 # 文章数
        sum_sim = 0     # 類似度の合計値
        average_sim = 0 # 類似度の平均値

        for num_row in d[num_d+1]:#1~5
            total_count+=1

            texts_vector_temp = df_candid_vector[df_candid_vector.index == num_row].vector_add_norm.tolist()[0]
            sum_sim += cos_sim(keywords_vector,texts_vector_temp)

        average_sim = sum_sim/total_count
        keywords_sim_list.append(average_sim)

        #print('=================')
        #print(keywords_sim_list)
        # ランキング付けをする
        #keywords_sim_list = number_ranking(keywords_sim_list)

        # ワードクラウドを作成する
        texts = []
        texts_wordcloud = ''

        for num_row in d[num_d+1]:#1~5
            texts_temp = df_candid_vector[df_candid_vector.index == num_row].text_morpho_list.tolist()[0]
            texts_temp = texts_temp.replace("'",'').replace('"','').replace('[','').replace(']','').replace(' ','').split(',')

            for remove_word in remove_words_list:
                try:
                    while remove_word in texts_temp:
                        texts_temp.remove(remove_word)
                except Exception as e:
                    c = 0

            texts.append(texts_temp)
            texts_wordcloud = texts_wordcloud+' '+' '.join(texts_temp)

        # wordcloud
        #print(texts_wordcloud)
        text = texts_wordcloud

        for stop_word in stop_word_list_wordcloud:
            text=text.replace(stop_word,'')

        not_search_words_morpho=words_morpho(not_search_words)

        for stop_word in not_search_words_morpho:
            text=text.replace(stop_word,'')

        if len(text)==0:
            text = 'not words'

#ワードクラウド

        wordcloud = WordCloud(background_color="white",font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", width=int(150*size_df_list_norm[num_d]),height=int(100*size_df_list_norm[num_d])).generate(text)

        wordcloud_extend = WordCloud(background_color="white",font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", width=300,height=200).generate(text)

        wordcloud.to_file("../../data/img_word_cloud/wordcloud_sample_"+ str(num_d+1)+".png")

        wordcloud_extend.to_file("../../data/img_word_cloud_extend/wordcloud_sample_"+ str(num_d+1)+".png")

        # 単語カウント
        # 単語を数える辞書を作成
        words = {}

        # split()でスペースと改行で分割したリストから単語を取り出す
        for word in text.split():
            # 単語をキーとして値に1を足していく。
            # 辞書に単語がない、すなわち初めて辞書に登録するときは0+1になる。
            words[word] = words.get(word, 0) + 1  #

        # リストに取り出して単語の出現回数でソート
        dic_freq = [(v, k) for k, v in words.items()]
        dic_freq.sort()
        dic_freq.reverse()

        list_freq.append(dic_freq)

    #print('=================')
    #print(keywords_sim_list)

    # ランキング付けをする
    keywords_sim_rank_list = []
    keywords_sim_rank_list = number_ranking(keywords_sim_list)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return df_candid_vector, d, list_freq, keywords_sim_rank_list


#ベクトル化をして重心を求める関数４４４４４４４４４
def vector_text_weight(df_candid_vector, input_words, not_search_words):
    start = time.time()
    docs_list_candid = []
    df_candid_vector = df_candid_vector
    df_candid_vector_list_temp = df_candid_vector.text_morpho_list.tolist()
    #=====================================================================
    # 候補条文のベクトル化
    #=====================================================================
    for i in range(len(df_candid_vector)):
        list_temp = df_candid_vector_list_temp[i].replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')
        docs_list_candid.append(' '.join(list_temp))

    # ベクトル化
    count_candid = CountVectorizer()
    docs_candid = np.array(docs_list_candid)
    bag_candid = count_candid.fit_transform(docs_candid)

    # ベクトルの要素の辞書
    dict_voca_num_candid = count_candid.vocabulary_
    dict_voca_num_candid_swap = {v: k for k, v in dict_voca_num_candid.items()}
    vector_list_candid = []

    for i in range(len(bag_candid.toarray())):
        vector_list_candid.append(list(bag_candid.toarray()[i]))

    df_candid_vector['vector'] = vector_list_candid
    #=====================================================================
    # ベクトルの正規化
    #=====================================================================
    # 正規化したvectorをvector_normに追加
    vector_norm_list = []

    for num_df in range(len(df_candid_vector)):
        # vectorを文章の長さで正規化する
        length=len(df_candid_vector.text_morpho_list.tolist()[num_df])

        #print()
        #print(length)
        #print(type(df_candid_vector['vector'][num_df]))
        #print(df_candid_vector['vector'][num_df])
        #print('===============')

        vector_norm = list(map(lambda x: x / length, df_candid_vector['vector'][num_df]))
        vector_norm_list.append(vector_norm)
        vector_norm = []

    df_candid_vector['vector_norm'] = vector_norm_list

    #=====================================================================
    # 継承候補の取得
    #=====================================================================

    id_reason_list = []

    for num_df in range(len(df_candid_vector)):
        id_reason_list_temp = []
        #for num in df_candid_vector['根拠_id_all'][num_df]:
        for num in df_candid_vector['根拠_id_all'][num_df]:
            if num != "''" and num != '0' and num != '' and len(num) != 0:
                #print(num)
                num = num.replace("'",'').replace('"','').replace('[','').replace(']','').strip(' ').strip("'").strip(' ')
                #print(num)
                if len(num) != 0 and num != ',':
                    id_reason_list_temp.append(int(num))
            else:
                id_reason_list_temp.append(0)

        #print(num_df)
        #for num_belong in df_candid_vector['所属'][num_df]:
        #print(num_belong)
        #print(df_candid_vector['所属'][num_df])

        if str(df_candid_vector['所属'][num_df])!='nan':
            num_belong = int(df_candid_vector['所属'][num_df])
            #print(num_belong)
            id_reason_list_temp.append(num_belong)


        id_reason_list.append(id_reason_list_temp)


    df_candid_vector['id_reason_list_for_vector'] = id_reason_list



    #=====================================================================
    # 継承ベクトルの生成
    #=====================================================================

    # 根拠を継承したベクトルを作成
    #　'vector_add_norm'に保存

    original_vector_norm_and_norm_list = []


    for num_df in range(len(df_candid_vector)):

        # add vector_norm
        original_vector_norm = df_candid_vector.vector_norm[num_df]

        # 追加するベクトルの総数
        try:
            if df_candid_vector.id_reason_list_for_vector[num_df][0]!= 0:
                sum_vector = len(df_candid_vector.id_reason_list_for_vector[num_df]) + 1
            else:
                sum_vector = 1
        except Exception as e:
            sum_vector = 1
        try:

            if df_candid_vector.id_reason_list_for_vector[num_df][0]!= 0:
                # 正規化したベクトルを足し合わせて、足し合わせたのちにベクトルの総数で割る
                for num_vector_each in range(len(df_candid_vector.id_reason_list_for_vector[num_df])):
                    vector_temp = df_candid_vector[df_candid_vector.id == df_candid_vector.id_reason_list_for_vector[num_df][num_vector_each]]['vector_norm'].tolist()

                    if len(vector_temp) == 0:
                        continue
                    #print(vector_temp)
                    # vector_norm add
                    original_vector_norm = [x + y for (x, y) in zip(original_vector_norm, vector_temp)]

                #print(num_df)
                #c = original_vector_norm
                if str(type(original_vector_norm[0])) == "<class 'numpy.ndarray'>":
                    # np.arrayをlistにキャスト
                    original_vector_norm = list(original_vector_norm[0])

        except Exception as e:
            sum_vector = 1

        # 足し合わせたベクトルをベクトルの総数で割る
        original_vector_norm_and_norm = list(map(lambda x: x / sum_vector, original_vector_norm))


        # 入力キーワードによって、ベクトルの重みを変更する
        # input_words: 最大値の'1.0'にする
        # not_search_words: 最小値の'0.0'にする

        input_words_morpho=words_morpho(input_words)
        not_search_words_morpho=words_morpho(not_search_words)

        input_word_vector_num_list = []
        for input_word in input_words_morpho:
            try:
                input_word_vector_num_list.append(dict_voca_num_candid[input_word])
            except:
                a=0

        not_search_word_vector_num_list = []
        for not_search_word in not_search_words_morpho:
            try:
                not_search_word_vector_num_list.append(dict_voca_num_candid[not_search_word])
            except:
                a=0
#値は調節する
        for not_search_index in not_search_word_vector_num_list:
            original_vector_norm_and_norm[not_search_index]= -1.0

        for input_word_index in input_word_vector_num_list:
            original_vector_norm_and_norm[input_word_index] = 1.0

        original_vector_norm_and_norm_list.append(original_vector_norm_and_norm)



    df_candid_vector['vector_add_norm'] = original_vector_norm_and_norm_list


    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return df_candid_vector, dict_voca_num_candid,dict_voca_num_candid_swap


# リスト内の値を大きいものから順に順位をつける５５５５５

# 順位のリストを返す
def number_ranking(closeList=[]):
    start = time.time()
    u, inv, counts = np.unique(closeList, return_inverse=True, return_counts=True)
    uniqueRankNd = np.array(np.hstack((0, counts[:-1].cumsum())), dtype='float16')
    uniqueRankNd = (counts == 1) * uniqueRankNd + (counts != 1) * (2 * uniqueRankNd + counts - 1) / counts
    numberRankNd = np.ones_like(inv) * inv.shape[0] - uniqueRankNd[inv]
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return list(map(int, numberRankNd))
