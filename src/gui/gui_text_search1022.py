import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon,QPixmap
from PyQt5.QtCore import pyqtSlot,Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QScrollArea,QWidget,QScrollArea,QVBoxLayout,QGroupBox,QLabel,QPushButton,QFormLayout,QComboBox,QTreeView,QSizePolicy
from PyQt5.Qt import QStandardItemModel, QStandardItem
from PyQt5 import QtWidgets
#　必要なライブラリを読み込む
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
import re
import nltk
import codecs
import json
from random import randint
# xml スクレイピング
from bs4 import BeautifulSoup
import xml.etree.ElementTree as et
# 形態素解析
import sys
import MeCab
import MeCab as mc
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence,Doc2Vec,TaggedDocument
from gensim import models
# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from copy import copy
import jaconv
import gensim
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
# 因子分析
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
import pathlib
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from itertools import product
#警告文を非表示
warnings.filterwarnings('ignore')
# base.pyのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
# 自作モジュールインポート
from utils import read_file_autohandle, words_extension, search_functions3#,ui

#ウィンドウの作成
class MainWindow(QtWidgets.QWidget):
    def __init__(self,parent=None):#初期ウィンドウ作成
        super(MainWindow, self).__init__(parent)

        # 縦横のレイアウト
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)
        self.setGeometry(200, 50, 800, 500)#(右,下,横幅,縦幅)前二つが、表示位置。後ろ二つが窓の大きさ
        self.setWindowTitle("検索")#ウィンドウの名前

        #検索ワード
        self.textbox_name = QtWidgets.QLabel('<p><font size="3" color="#ffffff">検索ワード</font></p>', self)#ネーミング
        self.textbox_name.move(20, 0)#位置の設定
        self.textbox = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.textbox.move(20, 20)#位置の設定
        self.textbox.resize(280,40)#テキストノックス大きさ

        #除外ワード
        self.textbox_name2 = QtWidgets.QLabel('<p><font size="3" color="#ffffff">除外ワード</font></p>', self)#ネーミング
        self.textbox_name2.move(20, 80)#位置の設定
        self.textbox2 = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.textbox2.move(20, 100)#位置の設定
        self.textbox2.resize(280,40)#テキストノックス大きさ

        #拡張機能
        self.kakutyo_name = QtWidgets.QLabel('<p><font size="3" color="#ffffff">拡張機能</font></p>', self)#テキストボックス読み込み
        self.kakutyo_name.move(20, 160)#位置の設定
        # QComboBoxオブジェクトの作成
        self.kakutyo = 0
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("拡張しない")
        self.combo.addItem("拡張する")
        self.combo.move(20, 180)
        self.combo.resize(140,40)
        self.combo.activated[str].connect(self.onActivated)

        #文章検索ボタン作成
        #self.sen_name = QtWidgets.QLabel('<p><font size="3" color="#ffffff">検索ボタン</font></p>', self)#テキストボックス読み込み
        #self.sen_name.move(20, 240)#位置の設定
        self.sen_button = QPushButton("検索",self)
        self.sen_button.move(20,260)#位置の設定
        self.sen_button.setFixedSize(140,40)
        # connect button to function on_click
        self.sen_button.clicked.connect(self.on_click1)

        self.show()

    #拡張するかしないか
    def onActivated(self, text):
        if str(text) == '拡張する':
            self.kakutyo = 1
        else:
            self.kakutyo = 0
    @pyqtSlot()#よくわからん

    def on_click1(self):#クリック後のウィンドウ変移
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=basic_auth("neo4j", "handhara66"))
        session = driver.session()
        #ボタンを隠す
        self.textbox.hide()
        self.textbox2.hide()
        self.sen_button.hide()
        #self.sen_name.hide()
        #self.wordname.hide()
        self.textbox_name.hide()
        self.textbox_name2.hide()
        self.kakutyo_name.hide()
        self.combo.hide()

        input_words = self.textbox.text()
        not_search_words = self.textbox2.text()
        self.save_word = self.textbox.text()
        self.save_remove_word = self.textbox2.text()
        # file_path_search_tempを用いて候補になる条文のidを取得し、
        # リストで返すキーワードを受け取る

        self.input_words = words_extension.words_morpho(input_words)
        self.not_search_words = words_extension.words_morpho(not_search_words)

        if self.kakutyo == 0:
            self.input_words_morpho = self.input_words
            self.not_search_words_morpho = self.not_search_words
        else:
            self.input_words_morpho = words_extension.words_extension(input_words)
            self.input_words_morpho = str(self.input_words_morpho).replace("[","").replace("]","").replace("'","").split(",")
            self.not_search_words_morpho = words_extension.words_extension(not_search_words)
            self.not_search_words_morpho = str(self.not_search_words_morpho).replace("[","").replace("]","").replace("'","").split(",")

        print(str(self.input_words_morpho))
        print(str(self.not_search_words_morpho))

        # 必要な検索ワード
        self.search_box_name = QLabel('<p><font size="3" color="#ffffff">検索ワード</font></p>', self)
        self.grid.addWidget(self.search_box_name,0,0)#入力ワードを記入
        self.textbox1 = QtWidgets.QLineEdit(self)
        self.textbox1.setText(self.textbox.text())
        self.textbox1.move(20, 20)#位置の設定
        self.textbox1.setFixedSize(600,40)
        self.grid.addWidget(self.textbox1,1,0)

        # 必要な共起ワード
        self.kyoki_name = QLabel('<p><font size="3" color="#ffffff">予測単語</font></p>', self)
        self.grid.addWidget(self.kyoki_name,2,0)#入力ワードを記入
        self.kyoki_words = search_functions3.inverse_lookup(inputs_words = self.input_words)
        self.kyoki_name_text = QtWidgets.QLineEdit(self)
        self.kyoki_name_text.setText(str(self.kyoki_words).replace("{","").replace("}","").replace("'"," "))
        self.kyoki_name_text.move(20, 20)#位置の設定
        self.kyoki_name_text.setFixedSize(600,40)
        self.grid.addWidget(self.kyoki_name_text,3,0)


        # 不必要なワード
        self.not_search_box_name = QLabel('<p><font size="3" color="#ffffff">除外ワード</font></p>', self)
        self.search_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.not_search_box_name,4,0)
        self.textbox1_remove = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.textbox1_remove.setText(self.textbox2.text())
        self.grid.addWidget(self.textbox1_remove,5,0)
        self.textbox1_remove.move(20, 20)#位置の設定
        self.textbox1_remove.setFixedSize(600,40)

        #拡張機能
        self.kakutyo_name = QtWidgets.QLabel('<p><font size="3" color="#ffffff">拡張機能</font></p>', self)#テキストボックス読み込み
        self.kakutyo_name.move(20, 200)#位置の設定
        self.grid.addWidget(self.kakutyo_name,0,1)
        # QComboBoxオブジェクトの作成
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("拡張しない")
        self.combo.addItem("拡張する")
        self.combo.move(20, 220)
        self.combo.setFixedSize(120,20)
        # アイテムが選択されたらonActivated関数の呼び出し
        self.combo.activated[str].connect(self.onActivated)
        self.grid.addWidget(self.combo,1,1)

        # 検索の実行ボタン
        self.button_execute_text_search = QtWidgets.QPushButton("再検索", self)
        self.button_execute_text_search.move(200,80)#位置の設定
        self.button_execute_text_search.setFixedSize(100,40)
        self.button_execute_text_search.clicked.connect(self.listviewCheckChanged)
        self.button_execute_text_search.clicked.connect(self.on_click_next)
        self.grid.addWidget(self.button_execute_text_search,5,1)

        #単語の意味検索
        self.word_box_name = QLabel('<p><font size="3" color="#ffffff">不明単語</font></p>', self)
        self.word_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.word_box_name,6,0)
        self.wordbox = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.grid.addWidget(self.wordbox,7,0)
        self.wordbox.move(20, 20)#位置の設定
        self.wordbox.setFixedSize(600,40)

        # 検索の実行ボタン
        self.button_word_search = QtWidgets.QPushButton("単語意味表示", self)
        self.button_word_search.move(20,800)#位置の設定
        self.button_word_search.setFixedSize(120,40)
        self.button_word_search.clicked.connect(self.on_click_next)
        self.grid.addWidget(self.button_word_search,7,1)

        #文章検索
        self.sen_box_name = QLabel('<p><font size="3" color="#ffffff">文書ID</font></p>', self)
        self.sen_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.sen_box_name,8,0)
        self.sen_box = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.grid.addWidget(self.sen_box,9,0)
        self.sen_box.move(20, 20)#位置の設定
        self.sen_box.setFixedSize(600,40)

        # 文章検索の実行ボタン
        self.button_sen_search = QtWidgets.QPushButton("文章検索", self)
        self.button_sen_search.move(20,800)#位置の設定
        self.button_sen_search.setFixedSize(120,40)
        self.button_sen_search.clicked.connect(self.listviewCheckChanged)
        self.button_sen_search.clicked.connect(self.on_click_sentence_next)
        self.grid.addWidget(self.button_sen_search,9,1)

        #文章表示ボタン
        self.text_search_button1 = QtWidgets.QPushButton('文章探索実行', self)
        self.text_search_button1.move(20,80)
        self.text_search_button1.setFixedSize(200,40)
        # SubWindowへ移動
        self.text_search_button1.clicked.connect(self.listviewCheckChanged)
        self.text_search_button1.clicked.connect(self.makeWindow)
        self.grid.addWidget(self.text_search_button1,10,1)

        # 入力したテキストを受け取る
        #textboxValue = self.textbox.text()
        #拡張したテキストを受け取る
        #textboxValue = words_extension.words_morpho(textboxValue)
        file_path_process4 = ['../../src/gui/all/df_candid_vector.csv','../../src/gui/all/dict_voca_num_candid.npy','../../src/gui/all/original_vector_norm_and_norm_and_norm.npy','../../src/gui/all/df_candid_vector_vector.csv','../../src/gui/all/vector_norm.npy']
        #self.df=pd.read_csv(file_path_process4[0])
        open = np.load(file_path_process4[1])
        #self.df_candid_vector,self.dict_voca_num_candid,self.dict_voca_num_candid_swap = search_functions2.vector_text_weight(df_candid_vector = df,input_words=input_words, not_search_words=not_search_words)
        self.dict_voca_num_candid=dict(open)
        #self.dict_voca_num_candid_swap=pd.read_csv(file_path_process4[2])
        self.original_vector_norm_and_norm_and_norm=np.load(file_path_process4[2])
        self.df_candid_vector=pd.read_csv(file_path_process4[3])
        self.integ_cluster_doc_id = self.df_candid_vector.id.tolist()
        self.df_candid_vector_all = pd.DataFrame()
        self.df_candid_vector_all = copy(self.df_candid_vector_all)
        self.df_candid_vector = self.df_candid_vector[self.df_candid_vector.id.isin(self.integ_cluster_doc_id)]
        self.df_candid_vector = self.df_candid_vector.reset_index(drop=True)
        #self.dict_voca_num_candid_swap = MainWindow.dict_voca_num_candid_swap


        # キーワードベクトルの初期化
        self.keywords_vector =[0.0 for i in range(len(self.dict_voca_num_candid))]
        # キーワードベクトルの作成
        self.input_word_vector_num_list = []
        for input_word in self.input_words_morpho:
            try:
                self.input_word_vector_num_list.append(int(self.dict_voca_num_candid[input_word]))
            except:
                a=0
        self.not_search_word_vector_num_list = []
        for not_search_word in self.not_search_words_morpho:
            try:
                self.not_search_word_vector_num_list.append(int(self.dict_voca_num_candid[not_search_word]))
            except:
                a=0
        for not_search_index in self.not_search_word_vector_num_list:
            self.keywords_vector[not_search_index]= -1.0
            #print(self.keywords_vector[not_search_index])
        for input_word_index in self.input_word_vector_num_list:
            self.keywords_vector[input_word_index]=1.0
            #print(self.keywords_vector[input_word_index])

        # キーワードベクトルと文章の距離計算vector_norm、file_path_search_tempを用いて候補になる条文のidを取得し、リストで返すキーワードを受け取る
        original_vector_norm_and_norm_and_norm_list=[self.original_vector_norm_and_norm_and_norm[num_df]for num_df in range(len(self.df_candid_vector))]
        self.df_candid_vector['vector_final_norm']=original_vector_norm_and_norm_and_norm_list
        #データの並び替え
        #self.df_candid_vector=search_functions3.final(df_candid_vector=self.df_candid_vector,original_vector_norm_and_norm_and_norm=self.original_vector_norm_and_norm_and_norm)
        self.sim_list = [search_functions3.cos_sim(self.keywords_vector, self.df_candid_vector.vector_final_norm[num]) for num in range(len(self.df_candid_vector.vector_final_norm))]
        self.df_candid_vector['sim'] = self.sim_list
        self.df_candid_vector_descending = self.df_candid_vector.sort_values('sim', ascending=False)[:50]
        self.kouho_id_list=self.df_candid_vector_descending['id'].tolist()

        #neo4jによる検索
        self.kouho_bunsyo_net_list=[]
        self.kouho_zyo_net_list=[]
        self.kouho_syo_net_list=[]
        for i in range(10):
            p = self.kouho_id_list[i]
            print(p)
            self.kouho_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.kouho_bunsyo:
                i=str(i).split("'")
                self.kouho_bunsyo_net_list.append(i[1])
            self.kouho_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.kouho_zyo:
                i=str(i).split("'")
                self.kouho_zyo_net_list.append(i[1])
            self.kouho_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.kouho_syo:
                i=str(i).split("'")
                self.kouho_syo_net_list.append(i[1])
        #print(self.kouho_id_list)
        #print(self.kouho_bunsyo_net_list)
        #print(self.kouho_zyo_net_list)
        #print(self.kouho_syo_net_list)

        #チェックボタンの判定
        self.model = QStandardItemModel(self)
        self.kouho_item_list=[]
        self.SubformLayout = QtWidgets.QVBoxLayout()
        #self.SubformLayout.resize(self.SubformLayout.sizeHint().width(), self.SubformLayout.minimumHeight())
        #self.SubformLayout.setGeometry(self)
        self.SubgroupBox =QtWidgets.QListView()
        self.model.appendRow(QStandardItem("チェックした文章は保存されます"))
        self.model.appendRow(QStandardItem('【候補文章】'))
        for i in range(10):
            self.kouho_item = QStandardItem("ID:"+str(self.kouho_id_list[i])+"　タイトル:"+str(self.kouho_bunsyo_net_list[i])+"　条名:"+str(self.kouho_zyo_net_list[i])+"　章名:"+str(self.kouho_syo_net_list[i])+" ")
            self.kouho_item.setCheckable(True)
            self.kouho_item.setSelectable(False)
            self.kouho_item.setCheckState(Qt.Unchecked)
            self.kouho_item_list.append([self.kouho_id_list[i],self.kouho_bunsyo_net_list[i],self.kouho_zyo_net_list[i],self.kouho_syo_net_list[i]])
            self.model.appendRow(self.kouho_item)

        self.model.appendRow(QStandardItem('【根拠となる文章】'))
        self.model.appendRow(QStandardItem('【補足となる文章】'))
        self.model.appendRow(QStandardItem('【所属となる文章】'))
        self.model.appendRow(QStandardItem('【類似度の高い文章】'))
        self.model.appendRow(QStandardItem('【保存文章】'))
        self.model.appendRow(QStandardItem(''))
        self.model.appendRow(QStandardItem(''))
        self.hozon_list_new=[]
        self.konkyo_item_list=[]
        self.hosoku_item_list=[]
        self.syozoku_item_list=[]
        self.ruizi_item_list=[]

        self.SubgroupBox.setModel(self.model)
        self.SubgroupBox.setLayout(self.SubformLayout)
        self.Subscroll=QtWidgets.QScrollArea()
        self.Subscroll.setWidget(self.SubgroupBox)
        self.Subscroll.setWidgetResizable(True)
        self.Subscroll.setFixedHeight(500)
        self.grid.addWidget(self.Subscroll,10,0)

    def listviewCheckChanged(self):
        self.check_list=[]
        for index in range(self.model.rowCount()):
            if self.model.item(index).checkState() == Qt.Checked:
                if index not in self.check_list:
                    self.check_list.append(index)
        print("チェックボックスチェック")
        print(self.check_list)
        #ワードの受け渡し
        self.kouho_item_list2 = self.kouho_item_list
        self.konkyo_item_list2 = self.konkyo_item_list
        self.hosoku_item_list2 = self.hosoku_item_list
        self.syozoku_item_list2 = self.syozoku_item_list
        self.ruizi_item_list2 = self.ruizi_item_list
        self.hozon_list_new2 = self.hozon_list_new
        self.check_list2 = self.check_list

    #結果表示後の再検索のウィンドウ
    def on_click_next(self):
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=basic_auth("neo4j", "handhara66"))
        session = driver.session()
        try:
            self.text_search_button1.hide()
        except:
            a=0

        try:
            self.sim_score
        except:
            a=0

        self.text_search_button1.hide()

        try:
            self.textbox1.hide()
        except:
            a=0
        #self.button1.hide()
        try:
            self.textbox1_remove.hide()
        except:
            a=0
        try:
            self.button1_remove.hide()
        except:
            a=0
        self.kyoki_name.hide()
        self.kyoki_name_text.hide()
        self.kakutyo_name.hide()
        self.combo.hide()
        self.search_box_name.hide()
        self.not_search_box_name.hide()
        try:
            self.button_execute_text_search.hide()
        except:
            a=0

        try:
            self.text_search_button1_next.hide()
        except:
            a=0
        try:
            self.textbox_next.hide()
        except:
            a=0

        try:
            self.button_next.hide()
        except:
            a=0

        try:
            self.textbox_next_remove.hide()
        except:
            a=0

        try:
            self.button_next_remove.hide()
        except:
            a=0

        try:
            self.save_text_temp = self.textbox_next.text()
        except:
            a=0
        #if self.count_next==1:
            # 入力した検索ワードを一時的に保存するon_clickの情報
            #self.save_text_temp=self.textbox1.text()
            #self.save_text_remove_temp = self.textbox1_remove.text()
        #else:
            #self.save_text_temp = self.textbox_next.text()
            #self.save_text_remove_temp = self.textbox_next_remove.text()


        #単語の保存
        input_words = self.textbox1.text()
        not_search_words = self.textbox1_remove.text()
        self.save_word = self.textbox1.text()
        self.save_remove_word = self.textbox1_remove.text()
        # file_path_search_tempを用いて候補になる条文のidを取得し、
        # リストで返す
        #キーワードを受け取る
        self.input_words = words_extension.words_morpho(input_words)
        self.not_search_words = words_extension.words_morpho(not_search_words)

        if self.kakutyo == 0:
            self.input_words_morpho = self.input_words
            self.not_search_words_morpho = self.not_search_words
        else:
            self.input_words_morpho = words_extension.words_extension(input_words)
            self.input_words_morpho = str(self.input_words_morpho).replace("[","").replace("]","").replace("'","").split(",")
            self.not_search_words_morpho = words_extension.words_extension(not_search_words)
            self.not_search_words_morpho = str(self.not_search_words_morpho).replace("[","").replace("]","").replace("'","").split(",")

        print(str(self.input_words_morpho))
        print(str(self.not_search_words_morpho))

        # 必要な検索ワード
        self.search_box_name = QLabel('<p><font size="3" color="#ffffff">検索ワード</font></p>', self)
        self.grid.addWidget(self.search_box_name,0,0)#入力ワードを記入
        self.textbox1 = QtWidgets.QLineEdit(self)
        self.textbox1.setText(self.save_word)
        self.textbox1.move(20, 20)#位置の設定
        self.textbox1.setFixedSize(600,40)
        self.grid.addWidget(self.textbox1,1,0)

        # 必要な共起ワード
        self.kyoki_name = QLabel('<p><font size="3" color="#ffffff">予測単語</font></p>', self)
        self.grid.addWidget(self.kyoki_name,2,0)#入力ワードを記入
        self.kyoki_words = search_functions3.inverse_lookup(inputs_words = self.input_words)
        self.kyoki_name_text = QtWidgets.QLineEdit(self)
        self.kyoki_name_text.setText(str(self.kyoki_words).replace("{","").replace("}","").replace("'"," "))
        self.kyoki_name_text.move(20, 20)#位置の設定
        self.kyoki_name_text.setFixedSize(600,40)
        self.grid.addWidget(self.kyoki_name_text,3,0)
        # Create a button in the window
        #self.button_next = QPushButton("Search text", self)
        #self.button_next.move(20,80)
        # connect button to function on_click
        #self.button_next.clicked.connect(self.on_click_next)
        #self.grid.addWidget(self.button_next)

        # 不必要なワード
        self.not_search_box_name = QLabel('<p><font size="3" color="#ffffff">除外ワード</font></p>', self)
        self.search_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.not_search_box_name,4,0)
        self.textbox1_remove = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.textbox1_remove.setText(self.save_remove_word)
        self.grid.addWidget(self.textbox1_remove,5,0)
        self.textbox1_remove.move(20, 20)#位置の設定
        self.textbox1_remove.setFixedSize(600,40)

        #拡張機能
        self.kakutyo_name = QtWidgets.QLabel('<p><font size="3" color="#ffffff">拡張機能</font></p>', self)#テキストボックス読み込み
        self.kakutyo_name.move(20, 200)#位置の設定
        self.grid.addWidget(self.kakutyo_name,0,1)
        # QComboBoxオブジェクトの作成
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("拡張しない")
        self.combo.addItem("拡張する")
        self.combo.move(20, 220)
        self.combo.setFixedSize(120,20)
        # アイテムが選択されたらonActivated関数の呼び出し
        self.combo.activated[str].connect(self.onActivated)
        self.grid.addWidget(self.combo,1,1)

        # Create a button in the window
        #self.button_next_remove = QPushButton("Unnecessary text", self)
        #self.button_next_remove.move(20,80)
        # connect button to function on_click
        #self.button_next_remove.clicked.connect(self.on_click_next)
        #self.show()
        #self.grid.addWidget(self.button_next_remove)
        # 検索の実行ボタン
        self.button_word_search = QtWidgets.QPushButton("単語意味表示", self)
        self.button_word_search.move(20,800)#位置の設定
        self.button_word_search.setFixedSize(120,40)
        self.button_word_search.clicked.connect(self.on_click_next)
        self.grid.addWidget(self.button_word_search,7,1)

        #文章検索
        self.sen_box_name = QLabel('<p><font size="3" color="#ffffff">文書ID</font></p>', self)
        self.sen_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.sen_box_name,8,0)
        self.sen_box = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.grid.addWidget(self.sen_box,9,0)
        self.sen_box.move(20, 20)#位置の設定
        self.sen_box.setFixedSize(600,40)

        # 文章検索の実行ボタン
        self.button_sen_search = QtWidgets.QPushButton("文章検索", self)
        self.button_sen_search.move(20,800)#位置の設定
        self.button_sen_search.setFixedSize(120,40)
        self.button_sen_search.clicked.connect(self.listviewCheckChanged)
        self.button_sen_search.clicked.connect(self.on_click_sentence_next)
        self.grid.addWidget(self.button_sen_search,9,1)

        self.text_search_button1 = QtWidgets.QPushButton('文章探索実行', self)
        self.text_search_button1.move(20,80)
        self.text_search_button1.setFixedSize(200,40)
        # SubWindowへ移動
        self.text_search_button1.clicked.connect(self.listviewCheckChanged)
        self.text_search_button1.clicked.connect(self.makeWindow)
        self.grid.addWidget(self.text_search_button1,10,1)

        # 検索の実行ボタン
        self.button_execute_text_search = QtWidgets.QPushButton("再検索", self)
        self.button_execute_text_search.move(200,80)#位置の設定
        self.button_execute_text_search.setFixedSize(100,40)
        self.button_execute_text_search.clicked.connect(self.listviewCheckChanged)
        self.button_execute_text_search.clicked.connect(self.on_click_next)
        self.grid.addWidget(self.button_execute_text_search,5,1)
        #self.save_text_temp = self.textbox_next.text()
        #self.save_text_remove_temp = self.textbox_next_remove.text()
        #print('nextの検索ワード')
        #print(self.save_text_temp)
        #print('nextの不必要な検索ワード')
        #print(self.save_text_remove_temp)

        # 入力したテキストを受け取る
        #textboxValue = self.textbox.text()
        #拡張したテキストを受け取る
        #textboxValue = words_extension.words_morpho(textboxValue)
        file_path_process4 = ['../../src/gui/all/df_candid_vector.csv','../../src/gui/all/dict_voca_num_candid.npy','../../src/gui/all/original_vector_norm_and_norm_and_norm.npy','../../src/gui/all/df_candid_vector_vector.csv','../../src/gui/all/vector_norm.npy']
        open = np.load(file_path_process4[1])
        self.dict_voca_num_candid=dict(open)
        self.original_vector_norm_and_norm_and_norm=np.load(file_path_process4[2])
        self.df_candid_vector=pd.read_csv(file_path_process4[3])
        self.integ_cluster_doc_id = self.df_candid_vector.id.tolist()
        self.df_candid_vector_all = pd.DataFrame()
        self.df_candid_vector_all = copy(self.df_candid_vector_all)
        self.df_candid_vector = self.df_candid_vector[self.df_candid_vector.id.isin(self.integ_cluster_doc_id)]
        self.df_candid_vector = self.df_candid_vector.reset_index(drop=True)
        #self.dict_voca_num_candid_swap = MainWindow.dict_voca_num_candid_swap

        # キーワードベクトルの初期化
        self.keywords_vector =[0.0 for i in range(len(self.dict_voca_num_candid))]
        # キーワードベクトルの作成
        self.input_word_vector_num_list = []
        for input_word in self.input_words_morpho:
            try:
                self.input_word_vector_num_list.append(int(self.dict_voca_num_candid[input_word]))
            except:
                a=0
        self.not_search_word_vector_num_list = []
        for not_search_word in self.not_search_words_morpho:
            try:
                self.not_search_word_vector_num_list.append(int(self.dict_voca_num_candid[not_search_word]))
            except:
                a=0
        for not_search_index in self.not_search_word_vector_num_list:
            self.keywords_vector[not_search_index]= -1.0
        for input_word_index in self.input_word_vector_num_list:
            self.keywords_vector[input_word_index]=1.0
        # キーワードベクトルと文章の距離計算vector_norm、file_path_search_tempを用いて候補になる条文のidを取得し、リストで返すキーワードを受け取る
        original_vector_norm_and_norm_and_norm_list=[self.original_vector_norm_and_norm_and_norm[num_df]for num_df in range(len(self.df_candid_vector))]
        self.df_candid_vector['vector_final_norm']=original_vector_norm_and_norm_and_norm_list
        #self.df_candid_vector=search_functions3.final(df_candid_vector=self.df_candid_vector,original_vector_norm_and_norm_and_norm=self.original_vector_norm_and_norm_and_norm)
        self.sim_list = [search_functions3.cos_sim(self.keywords_vector, self.df_candid_vector.vector_final_norm[num]) for num in range(len(self.df_candid_vector.vector_final_norm))]
        self.df_candid_vector['sim'] = self.sim_list
        self.df_candid_vector_descending = self.df_candid_vector.sort_values('sim', ascending=False)[:10]
        self.kouho_id_list=self.df_candid_vector_descending['id'].tolist()

        self.hozon_list=[]
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.kouho_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.konkyo_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.hosoku_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.syozoku_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.ruizi_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.hozon_list_new)

        print(self.hozon_list)
        print("ここから候補文章")
        #候補文章提示
        #neo4jによる検索
        self.kouho_bunsyo_net_list=[]
        self.kouho_zyo_net_list=[]
        self.kouho_syo_net_list=[]
        for i in range(10):
            p = self.kouho_id_list[i]
            #print(p)
            self.kouho_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.kouho_bunsyo:
                i=str(i).split("'")
                self.kouho_bunsyo_result=0
                self.kouho_bunsyo_result=i[1]
                self.kouho_bunsyo_net_list.append(self.kouho_bunsyo_result)
            self.kouho_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.kouho_zyo:
                i=str(i).split("'")
                self.kouho_zyo_result=0
                self.kouho_zyo_result=i[1]
                self.kouho_zyo_net_list.append(self.kouho_zyo_result)
            self.kouho_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.kouho_syo:
                i=str(i).split("'")
                self.kouho_syo_result=0
                self.kouho_syo_result=i[1]
                self.kouho_syo_net_list.append(self.kouho_syo_result)
        #print(self.kouho_id_list)
        #print(self.kouho_bunsyo_net_list)
        #print(self.kouho_zyo_net_list)
        #print(self.kouho_syo_net_list)

        #チェックボタンの判定
        self.model = QStandardItemModel(self)
        self.SubformLayout=QtWidgets.QFormLayout()
        self.SubgroupBox =QtWidgets.QListView()
        self.model.appendRow(QStandardItem("チェックした文章は保存されます"))
        self.model.appendRow(QStandardItem("【候補文章】"))
        self.kouho_item_list=[]
        for i in range(10):
            self.kouho_item = QStandardItem("ID:"+str(self.kouho_id_list[i])+"　タイトル:"+str(self.kouho_bunsyo_net_list[i])+"　条名:"+str(self.kouho_zyo_net_list[i])+"　章名:"+str(self.kouho_syo_net_list[i])+" ")
            self.kouho_item.setCheckable(True)
            self.kouho_item.setSelectable(False)
            self.kouho_item.setCheckState(Qt.Unchecked)
            self.kouho_item_list.append([self.kouho_id_list[i],self.kouho_bunsyo_net_list[i],self.kouho_zyo_net_list[i],self.kouho_syo_net_list[i]])
            #print(self.kouho_item_list)
            self.model.appendRow(self.kouho_item)

        print("ここから根拠文章")
        self.model.appendRow(QStandardItem('【根拠となる文章】'))
        #根拠文章提示
        #neo4jによる検索
        self.research_id =0
        #print(self.research_id)
        self.konkyo_bunsyo_net_list=[]
        self.konkyo_zyo_net_list=[]
        self.konkyo_syo_net_list=[]
        self.konkyo_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`根拠`]-(m:sentence) RETURN m.`文書id`;");
        self.konkyo_id_list=[]
        for i in self.konkyo_bunsyo:
            i=str(i)
            self.konkyo_id=re.findall('=(.*)>',i)
            self.konkyo_id_list.append(self.konkyo_id)
        #print(self.konkyo_id_list)
        for p in self.konkyo_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.konkyo_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.konkyo_bunsyo:
                i=str(i).split("'")
                self.konkyo_bunsyo_result=0
                self.konkyo_bunsyo_result=i[1]
                self.konkyo_kouho_bunsyo_net_list.append(self.konkyo_bunsyo_result)
            self.konkyo_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.konkyo_zyo:
                i=str(i).split("'")
                self.konkyo_zyo_result=0
                self.konkyo_zyo_result=i[1]
                self.konkyo_zyo_net_list.append(self.konkyo_zyo_result)
            self.konkyo_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.konkyo_syo:
                i=str(i).split("'")
                self.konkyo_syo_result=0
                self.konkyo_syo_result=i[1]
                self.konkyo_syo_net_list.append(self.konkyo_syo_result)
        #print(self.konkyo_id_list)
        #print(self.konkyo_bunsyo_net_list)
        #print(self.konkyo_zyo_net_list)
        #print(self.konkyo_syo_net_list)
        self.konkyo_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.konkyo_id_list)):
            self.konkyo_item = QStandardItem("ID:"+str(str(self.konkyo_id_list[i].replace("[","").replace("]","").replace("'","")))+"　タイトル:"+str(self.konkyo_bunsyo_net_list[i])+"　条名:"+str(self.konkyo_zyo_net_list[i])+"　章名:"+str(self.konkyo_syo_net_list[i])+" ")
            self.konkyo_item.setCheckable(True)
            self.konkyo_item.setSelectable(False)
            self.konkyo_item.setCheckState(Qt.Unchecked)
            self.konkyo_item_list.append([self.konkyo_id_list[i],self.konkyo_bunsyo_net_list[i],self.konkyo_zyo_net_list[i],self.konkyo_syo_net_list[i]])
            print(self.konkyo_item_list)
            self.model.appendRow(self.konkyo_item)

        print("ここから補足文章")
        self.model.appendRow(QStandardItem('【補足となる文章】'))
        #補足する文章
        #neo4jによる検索
        self.hosoku_bunsyo_net_list=[]
        self.hosoku_zyo_net_list=[]
        self.hosoku_syo_net_list=[]
        self.hosoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`補足`]-(m:sentence) RETURN m.`文書id`;");
        self.hosoku_id_list=[]
        for i in self.hosoku_bunsyo:
            i=str(i)
            self.hosoku_id=re.findall('=(.*)>',i)
            self.hosoku_id_list.append(self.hosoku_id)
        #print(self.hosoku_id_list)
        for p in self.hosoku_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.hosoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.hosoku_bunsyo:
                i=str(i).split("'")
                self.hosoku_bunsyo_result=0
                self.hosoku_bunsyo_result=i[1]
                self.hosoku_kouho_bunsyo_net_list.append(self.hosoku_bunsyo_result)
            self.hosoku_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.hosoku_zyo:
                i=str(i).split("'")
                self.hosoku_zyo_result=0
                self.hosoku_zyo_result=i[1]
                self.hosoku_zyo_net_list.append(self.hosoku_zyo_result)
            self.hosoku_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.hosoku_syo:
                i=str(i).split("'")
                self.hosoku_syo_result=0
                self.hosoku_syo_result=i[1]
                self.hosoku_syo_net_list.append(self.hosoku_syo_result)
        #print(self.hosoku_id_list)
        #print(self.hosoku_bunsyo_net_list)
        #print(self.hosoku_zyo_net_list)
        #print(self.hosoku_syo_net_list)
        self.hosoku_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.hosoku_id_list)):
            self.hosoku_item = QStandardItem("ID:"+str(str(self.hosoku_id_list[i].replace("[","").replace("]","").replace("'","")))+"　タイトル:"+str(self.hosoku_bunsyo_net_list[i])+"　条名:"+str(self.hosoku_zyo_net_list[i])+"　章名:"+str(self.hosoku_syo_net_list[i])+" ")
            self.hosoku_item.setCheckable(True)
            self.hosoku_item.setSelectable(False)
            self.hosoku_item.setCheckState(Qt.Unchecked)
            self.hosoku_item_list.append([self.hosoku_id_list[i],self.hosoku_bunsyo_net_list[i],self.hosoku_zyo_net_list[i],self.hosoku_syo_net_list[i]])
            print(self.hosoku_item_list)
            self.model.appendRow(self.hosoku_item)

        print("ここから所属文章")
        self.model.appendRow(QStandardItem('【所属となる文章】'))
        #所属文書の検索
        self.syozoku_bunsyo_net_list=[]
        self.syozoku_zyo_net_list=[]
        self.syozoku_syo_net_list=[]
        self.syozoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`所属`]-(m:sentence) RETURN m.`文書id`;");
        self.syozoku_id_list=[]
        for i in self.syozoku_bunsyo:
            i=str(i)
            self.syozoku_id=re.findall('=(.*)>',i)
            self.syozoku_id_list.append(self.syozoku_id)
        #print(self.syozoku_id_list)
        for p in self.syozoku_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.syozoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.syozoku_bunsyo:
                i=str(i).split("'")
                self.syozoku_bunsyo_result=0
                self.syozoku_bunsyo_result=i[1]
                self.syozoku_kouho_bunsyo_net_list.append(self.syozoku_bunsyo_result)
            self.syozoku_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.syozoku_zyo:
                i=str(i).split("'")
                self.syozoku_zyo_result=0
                self.syozoku_zyo_result=i[1]
                self.syozoku_zyo_net_list.append(self.syozoku_zyo_result)
            self.syozoku_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.syozoku_syo:
                i=str(i).split("'")
                self.syozoku_syo_result=0
                self.syozoku_syo_result=i[1]
                self.syozoku_syo_net_list.append(self.syozoku_syo_result)
        #print(self.syozoku_id_list)
        #print(self.syozoku_bunsyo_net_list)
        #print(self.syozoku_zyo_net_list)
        #print(self.syozoku_syo_net_list)
        self.syozoku_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.syozoku_id_list)):
            self.syozoku_item = QStandardItem("ID:"+str(str(self.syozoku_id_list[i].replace("[","").replace("]","").replace("'","")))+"　タイトル:"+str(self.syozoku_bunsyo_net_list[i])+"　条名:"+str(self.syozoku_zyo_net_list[i])+"　章名:"+str(self.syozoku_syo_net_list[i])+" ")
            self.syozoku_item.setCheckable(True)
            self.syozoku_item.setSelectable(False)
            self.syozoku_item.setCheckState(Qt.Unchecked)
            self.syozoku_item_list.append([self.syozoku_id_list[i],self.syozoku_bunsyo_net_list[i],self.syozoku_zyo_net_list[i],self.syozoku_syo_net_list[i]])
            print(self.syozoku_item_list)
            self.model.appendRow(self.syozoku_item)


        print("ここから類似文章")
        self.model.appendRow(QStandardItem('【類似度の高い文章】'))
        #類似文章
        #neo4jによる検索
        self.ruizi_bunsyo_net_list=[]
        self.ruizi_zyo_net_list=[]
        self.ruizi_syo_net_list=[]
        self.ruizi_do_net_list =[]
        self.ruizi_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`類似`]-(m:sentence) RETURN m.`文書id`;");
        self.ruizi_id_list=[]

        for i in self.ruizi_bunsyo:
            i=str(i)
            self.ruizi_id=re.findall('=(.*)>',i)
            self.ruizi_id_list.append(self.ruizi_id)
        #print(self.ruizi_id_list)
        for p in self.ruizi_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.ruizi_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.ruizi_bunsyo:
                i=str(i).split("'")
                self.ruizi_bunsyo_result=0
                self.ruizi_bunsyo_result=i[1]
                self.ruizi_bunsyo_net_list.append(self.ruizi_bunsyo_result)
            self.ruizi_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.ruizi_zyo:
                i=str(i).split("'")
                self.ruizi_zyo_result=0
                self.ruizi_zyo_result=i[1]
                self.ruizi_zyo_net_list.append(self.ruizi_zyo_result)
            self.ruizi_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.ruizi_syo:
                i=str(i).split("'")
                self.ruizi_syo_result=0
                self.ruizi_syo_result=i[1]
                self.ruizi_syo_net_list.append(self.ruizi_syo_result)
            self.ruizi_do=session.run("MATCH (n:sentence{文書id:"+ str(p)+"})-[r:`類似`]-(m:sentence{文書id:"+str(self.research_id)+"}) RETURN r.類似度;");
            for i in self.ruizi_do:
                i=str(i)
                self.ruizi_do_result=0
                self.ruizi_do_result=re.findall('=(.*)>',i)
                self.ruizi_do_net_list.append(self.ruizi_do_result)
        #print(len(self.ruizi_id_list))
        #print(len(self.ruizi_bunsyo_net_list))
        #print(len(self.ruizi_zyo_net_list))
        #print(len(self.ruizi_syo_net_list))
        #print(len(self.ruizi_do_net_list))
        self.ruizi_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.ruizi_id_list)):
            self.ruizi_item = QStandardItem("ID:"+str(str(self.ruizi_id_list[i].replace("[","").replace("]","").replace("'","")))+"　タイトル:"+str(self.ruizi_bunsyo_net_list[i])+" 条名:"+str(self.ruizi_zyo_net_list[i])+"　章名:"+str(self.ruizi_syo_net_list[i])+"　類似度:"+str(self.ruizi_do_net_list[i])+" ")
            self.ruizi_item.setCheckable(True)
            self.ruizi_item.setSelectable(False)
            self.ruizi_item.setCheckState(Qt.Unchecked)
            self.ruizi_item_list.append([self.ruizi_id_list[i],self.ruizi_bunsyo_net_list[i],self.ruizi_zyo_net_list[i],self.ruizi_syo_net_list[i]])
            print(self.ruizi_item_list)
            self.model.appendRow(self.ruizi_item)

        print("ここから保存文章")
        #チェックボックスに入れられた文章を保存
        self.model.appendRow(QStandardItem('【保存文章】'))
        #チェックボックスの判断
        self.hozon_list_new =[]
        self.check_check_list=[]
        for i in range(len(self.hozon_list)):
            if i in self.check_list:
                print(i)
                self.check_check_list.append(i)
                self.check_number=0
                for l in self.check_check_list:
                    if self.hozon_list[i][0]==self.hozon_list[l][0]:
                        self.check_number=self.check_number+1
                if self.check_number==1:
                    self.hozon_item = QStandardItem("ID:"+str(str(self.hozon_list[i][0]).replace("[","").replace("]","").replace("'",""))+"　タイトル:"+str(self.hozon_list[i][1])+" 条名:"+str(self.hozon_list[i][2])+"　章名:"+str(self.hozon_list[i][2])+" ")
                    self.hozon_item.setCheckable(True)
                    self.hozon_item.setSelectable(False)
                    self.hozon_item.setCheckState(Qt.Checked)
                    self.hozon_list_new.append([self.hozon_list[i][0],self.hozon_list[i][1],self.hozon_list[i][2],self.hozon_list[i][3]])
                    self.model.appendRow(self.hozon_item)

        self.SubgroupBox.setModel(self.model)
        self.SubgroupBox.setLayout(self.SubformLayout)
        self.Subscroll=QtWidgets.QScrollArea()
        self.Subscroll.setWidget(self.SubgroupBox)
        self.Subscroll.setWidgetResizable(True)
        self.Subscroll.setFixedHeight(500)
        self.grid.addWidget(self.Subscroll,10,0)
        self.model.appendRow(QStandardItem(''))
        self.model.appendRow(QStandardItem(''))


    def on_click_sentence_next(self):
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=basic_auth("neo4j", "handhara66"))
        session = driver.session()

        try:
            self.text_search_button1.hide()
        except:
            a=0

        try:
            self.sim_score
        except:
            a=0

        self.text_search_button1.hide()

        try:
            self.textbox1.hide()
        except:
            a=0
        #self.button1.hide()
        try:
            self.textbox1_remove.hide()
        except:
            a=0
        try:
            self.button1_remove.hide()
        except:
            a=0
        self.kyoki_name.hide()
        self.kyoki_name_text.hide()
        self.kakutyo_name.hide()
        self.combo.hide()
        self.search_box_name.hide()
        self.not_search_box_name.hide()

        try:
            self.button_execute_text_search.hide()
        except:
            a=0

        try:
            self.text_search_button1_next.hide()
        except:
            a=0

        try:
            self.textbox_next.hide()
        except:
            a=0

        try:
            self.button_next.hide()
        except:
            a=0

        try:
            self.textbox_next_remove.hide()
        except:
            a=0

        try:
            self.button_next_remove.hide()
        except:
            a=0

        try:
            self.save_text_temp = self.textbox_next.text()
        except:
            a=0

        #if self.count_next==1:
            # 入力した検索ワードを一時的に保存するon_clickの情報
            #self.save_text_temp=self.textbox1.text()
            #self.save_text_remove_temp = self.textbox1_remove.text()
        #else:
            #self.save_text_temp = self.textbox_next.text()
            #self.save_text_remove_temp = self.textbox_next_remove.text()

        #単語の保存
        input_words = self.save_word
        not_search_words = self.save_remove_word
        #self.save_word = self.textbox1.text()
        #self.save_remove_word = self.textbox1_remove.text()
        self.save_id = self.sen_box.text()
        # file_path_search_tempを用いて候補になる条文のidを取得し、
        # リストで返す
        #キーワードを受け取る
        self.input_words = words_extension.words_morpho(input_words)
        self.not_search_words = words_extension.words_morpho(not_search_words)

        if self.kakutyo == 0:
            self.input_words_morpho = self.input_words
            self.not_search_words_morpho = self.not_search_words
        else:
            self.input_words_morpho = words_extension.words_extension(input_words)
            self.input_words_morpho = str(self.input_words_morpho).replace("[","").replace("]","").replace("'","").split(",")
            self.not_search_words_morpho = words_extension.words_extension(not_search_words)
            self.not_search_words_morpho = str(self.not_search_words_morpho).replace("[","").replace("]","").replace("'","").split(",")

        print(str(self.input_words_morpho))
        print(str(self.not_search_words_morpho))

        # 必要な検索ワード
        self.search_box_name = QLabel('<p><font size="3" color="#ffffff">検索ワード</font></p>', self)
        self.grid.addWidget(self.search_box_name,0,0)#入力ワードを記入
        self.textbox1 = QtWidgets.QLineEdit(self)
        self.textbox1.setText(self.save_word)
        self.textbox1.move(20, 20)#位置の設定
        self.textbox1.setFixedSize(600,40)
        self.grid.addWidget(self.textbox1,1,0)

        # 必要な共起ワード
        self.kyoki_name = QLabel('<p><font size="3" color="#ffffff">予測単語</font></p>', self)
        self.grid.addWidget(self.kyoki_name,2,0)#入力ワードを記入
        self.kyoki_words = search_functions3.inverse_lookup(inputs_words = self.input_words)
        self.kyoki_name_text = QtWidgets.QLineEdit(self)
        self.kyoki_name_text.setText(str(self.kyoki_words).replace("{","").replace("}","").replace("'"," "))
        self.kyoki_name_text.move(20, 20)#位置の設定
        self.kyoki_name_text.setFixedSize(600,40)
        self.grid.addWidget(self.kyoki_name_text,3,0)
        # Create a button in the window
        #self.button_next = QPushButton("Search text", self)
        #self.button_next.move(20,80)
        # connect button to function on_click
        #self.button_next.clicked.connect(self.on_click_next)
        #self.grid.addWidget(self.button_next)

        # 不必要なワード
        self.not_search_box_name = QLabel('<p><font size="3" color="#ffffff">除外ワード</font></p>', self)
        self.search_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.not_search_box_name,4,0)
        self.textbox1_remove = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.textbox1_remove.setText(self.save_remove_word)
        self.grid.addWidget(self.textbox1_remove,5,0)
        self.textbox1_remove.move(20, 20)#位置の設定
        self.textbox1_remove.setFixedSize(600,40)

        #拡張機能
        self.kakutyo_name = QtWidgets.QLabel('<p><font size="3" color="#ffffff">拡張機能</font></p>', self)#テキストボックス読み込み
        self.kakutyo_name.move(20, 200)#位置の設定
        self.grid.addWidget(self.kakutyo_name,0,1)
        # QComboBoxオブジェクトの作成
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("拡張しない")
        self.combo.addItem("拡張する")
        self.combo.move(20, 220)
        self.combo.setFixedSize(120,20)
        # アイテムが選択されたらonActivated関数の呼び出し
        self.combo.activated[str].connect(self.onActivated)
        self.grid.addWidget(self.combo,1,1)

        # Create a button in the window
        #self.button_next_remove = QPushButton("Unnecessary text", self)
        #self.button_next_remove.move(20,80)
        # connect button to function on_click
        #self.button_next_remove.clicked.connect(self.on_click_next)
        #self.show()
        #self.grid.addWidget(self.button_next_remove)
        # 検索の実行ボタン
        self.button_word_search = QtWidgets.QPushButton("単語意味表示", self)
        self.button_word_search.move(20,800)#位置の設定
        self.button_word_search.setFixedSize(120,40)
        self.button_word_search.clicked.connect(self.on_click_next)
        self.grid.addWidget(self.button_word_search,7,1)

        #文章検索
        self.sen_box_name = QLabel('<p><font size="3" color="#ffffff">文書ID</font></p>', self)
        self.sen_box_name.move(20, 60)#位置の設定
        self.grid.addWidget(self.sen_box_name,8,0)
        self.sen_box = QtWidgets.QLineEdit(self)#テキストボックス読み込み
        self.grid.addWidget(self.sen_box,9,0)
        self.sen_box.move(20, 20)#位置の設定
        self.sen_box.setFixedSize(600,40)

        # 文章検索の実行ボタン
        self.button_sen_search = QtWidgets.QPushButton("文章検索", self)
        self.button_sen_search.move(20,800)#位置の設定
        self.button_sen_search.setFixedSize(120,40)
        self.button_sen_search.clicked.connect(self.listviewCheckChanged)
        self.button_sen_search.clicked.connect(self.on_click_sentence_next)
        self.grid.addWidget(self.button_sen_search,9,1)

        self.text_search_button1 = QtWidgets.QPushButton('文章探索実行', self)
        self.text_search_button1.move(20,80)
        self.text_search_button1.setFixedSize(200,40)
        # SubWindowへ移動
        self.text_search_button1.clicked.connect(self.listviewCheckChanged)
        self.text_search_button1.clicked.connect(self.makeWindow)
        self.grid.addWidget(self.text_search_button1,10,1)


        # 検索の実行ボタン
        self.button_execute_text_search = QtWidgets.QPushButton("再検索", self)
        self.button_execute_text_search.move(200,80)#位置の設定
        self.button_execute_text_search.setFixedSize(100,40)
        self.button_execute_text_search.clicked.connect(self.listviewCheckChanged)
        self.button_execute_text_search.clicked.connect(self.on_click_next)
        self.grid.addWidget(self.button_execute_text_search,5,1)

        #self.save_text_temp = self.textbox_next.text()
        #self.save_text_remove_temp = self.textbox_next_remove.text()
        #print('nextの検索ワード')
        #print(self.save_text_temp)
        #print('nextの不必要な検索ワード')
        #print(self.save_text_remove_temp)

        # 入力したテキストを受け取る
        #textboxValue = self.textbox.text()
        #拡張したテキストを受け取る
        #textboxValue = words_extension.words_morpho(textboxValue)
        file_path_process4 = ['../../src/gui/all/df_candid_vector.csv','../../src/gui/all/dict_voca_num_candid.npy','../../src/gui/all/original_vector_norm_and_norm_and_norm.npy','../../src/gui/all/df_candid_vector_vector.csv','../../src/gui/all/vector_norm.npy']

        open = np.load(file_path_process4[1])
        self.dict_voca_num_candid=dict(open)
        self.original_vector_norm_and_norm_and_norm=np.load(file_path_process4[2])
        self.df_candid_vector=pd.read_csv(file_path_process4[3])
        self.integ_cluster_doc_id = self.df_candid_vector.id.tolist()
        self.df_candid_vector_all = pd.DataFrame()
        self.df_candid_vector_all = copy(self.df_candid_vector_all)
        self.df_candid_vector = self.df_candid_vector[self.df_candid_vector.id.isin(self.integ_cluster_doc_id)]
        self.df_candid_vector = self.df_candid_vector.reset_index(drop=True)
        #self.dict_voca_num_candid_swap = MainWindow.dict_voca_num_candid_swap

        # キーワードベクトルの初期化
        self.keywords_vector =[0.0 for i in range(len(self.dict_voca_num_candid))]
        # キーワードベクトルの作成
        self.input_word_vector_num_list = []
        for input_word in self.input_words_morpho:
            try:
                self.input_word_vector_num_list.append(int(self.dict_voca_num_candid[input_word]))
            except:
                a=0
        self.not_search_word_vector_num_list = []
        for not_search_word in self.not_search_words_morpho:
            try:
                self.not_search_word_vector_num_list.append(int(self.dict_voca_num_candid[not_search_word]))
            except:
                a=0
        for not_search_index in self.not_search_word_vector_num_list:
            self.keywords_vector[not_search_index]= -1.0
            #print(self.keywords_vector[not_search_index])
        for input_word_index in self.input_word_vector_num_list:
            self.keywords_vector[input_word_index]=1.0
            #print(self.keywords_vector[input_word_index])
        # キーワードベクトルと文章の距離計算vector_norm
        # file_path_search_tempを用いて候補になる条文のidを取得し、リストで返すキーワードを受け取る
        original_vector_norm_and_norm_and_norm_list=[self.original_vector_norm_and_norm_and_norm[num_df]for num_df in range(len(self.df_candid_vector))]
        self.df_candid_vector['vector_final_norm']=original_vector_norm_and_norm_and_norm_list

        #self.df_candid_vector=search_functions3.final(df_candid_vector=self.df_candid_vector,original_vector_norm_and_norm_and_norm=self.original_vector_norm_and_norm_and_norm)
        self.sim_list = [search_functions3.cos_sim(self.keywords_vector, self.df_candid_vector.vector_final_norm[num]) for num in range(len(self.df_candid_vector.vector_final_norm))]
        self.df_candid_vector['sim'] = self.sim_list
        self.kouho_id_list=self.df_candid_vector_descending['id'].tolist()

        self.hozon_list=[]
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.kouho_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.konkyo_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.hosoku_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.syozoku_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.ruizi_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.hozon_list_new)

        #候補文章提示
        #neo4jによる検索
        self.kouho_bunsyo_net_list=[]
        self.kouho_zyo_net_list=[]
        self.kouho_syo_net_list=[]
        for i in range(10):
            p = self.kouho_id_list[i]
            #print(p)
            self.kouho_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.kouho_bunsyo:
                i=str(i).split("'")
                self.kouho_bunsyo_result=0
                self.kouho_bunsyo_result=i[1]
                self.kouho_bunsyo_net_list.append(self.kouho_bunsyo_result)
            self.kouho_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.kouho_zyo:
                i=str(i).split("'")
                self.kouho_zyo_result=0
                self.kouho_zyo_result=i[1]
                self.kouho_zyo_net_list.append(self.kouho_zyo_result)
            self.kouho_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.kouho_syo:
                i=str(i).split("'")
                self.kouho_syo_result=0
                self.kouho_syo_result=i[1]
                self.kouho_syo_net_list.append(self.kouho_syo_result)
        #print(self.kouho_id_list)
        #print(self.kouho_bunsyo_net_list)
        #print(self.kouho_zyo_net_list)
        #print(self.kouho_syo_net_list)

        #チェックボタンの判定
        self.model = QStandardItemModel(self)
        self.SubformLayout=QtWidgets.QFormLayout()
        #self.SubformLayout.resize(self.SubformLayout.sizeHint().width(), self.SubformLayout.minimumHeight())
        #self.SubformLayout.setGeometry(self)
        self.SubgroupBox =QtWidgets.QListView()
        self.model.appendRow(QStandardItem("チェックした文章は保存されます"))
        self.model.appendRow(QStandardItem("【候補文章】"))
        self.kouho_item_list=[]
        for i in range(10):
            self.kouho_item = QStandardItem("ID:"+str(self.kouho_id_list[i])+"　タイトル:"+str(self.kouho_bunsyo_net_list[i])+"　条名:"+str(self.kouho_zyo_net_list[i])+"　章名:"+str(self.kouho_syo_net_list[i])+" ")
            self.kouho_item.setCheckable(True)
            self.kouho_item.setSelectable(False)
            self.kouho_item.setCheckState(Qt.Unchecked)
            self.kouho_item_list.append([self.kouho_id_list[i],self.kouho_bunsyo_net_list[i],self.kouho_zyo_net_list[i],self.kouho_syo_net_list[i]])
            #print(self.kouho_item_list)
            self.model.appendRow(self.kouho_item)


        print("ここから根拠文章")
        self.model.appendRow(QStandardItem('【根拠となる文章】'))
        #根拠文章提示
        #neo4jによる検索
        self.research_id =0
        try:
            if int(self.save_id)>0:
                self.research_id = self.save_id
        except:
            a=0
        print(self.research_id)
        self.konkyo_bunsyo_net_list=[]
        self.konkyo_zyo_net_list=[]
        self.konkyo_syo_net_list=[]
        self.konkyo_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`根拠`]-(m:sentence) RETURN m.`文書id`;");
        self.konkyo_id_list=[]
        for i in self.konkyo_bunsyo:
            i=str(i)
            self.konkyo_id=re.findall('=(.*)>',i)
            self.konkyo_id_list.append(self.konkyo_id)
        #print(self.konkyo_id_list)
        for p in self.konkyo_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.konkyo_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.konkyo_bunsyo:
                i=str(i).split("'")
                self.konkyo_bunsyo_result=0
                self.konkyo_bunsyo_result=i[1]
                self.konkyo_bunsyo_net_list.append(self.konkyo_bunsyo_result)
            self.konkyo_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.konkyo_zyo:
                i=str(i).split("'")
                self.konkyo_zyo_result=0
                self.konkyo_zyo_result=i[1]
                self.konkyo_zyo_net_list.append(self.konkyo_zyo_result)
            self.konkyo_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.konkyo_syo:
                i=str(i).split("'")
                self.konkyo_syo_result=0
                self.konkyo_syo_result=i[1]
                self.konkyo_syo_net_list.append(self.konkyo_syo_result)
        #print(self.konkyo_id_list)
        #print(self.konkyo_bunsyo_net_list)
        #print(self.konkyo_zyo_net_list)
        #print(self.konkyo_syo_net_list)
        self.konkyo_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.konkyo_id_list)):
            self.konkyo_item = QStandardItem("ID:"+str(str(self.konkyo_id_list[i]).replace("[","").replace("]","").replace("'",""))+"　タイトル:"+str(self.konkyo_bunsyo_net_list[i])+"　条名:"+str(self.konkyo_zyo_net_list[i])+"　章名:"+str(self.konkyo_syo_net_list[i])+" ")
            self.konkyo_item.setCheckable(True)
            self.konkyo_item.setSelectable(False)
            self.konkyo_item.setCheckState(Qt.Unchecked)
            self.konkyo_item_list.append([self.konkyo_id_list[i],self.konkyo_bunsyo_net_list[i],self.konkyo_zyo_net_list[i],self.konkyo_syo_net_list[i]])
            print(self.konkyo_item_list)
            self.model.appendRow(self.konkyo_item)


        print("ここから補足文章")
        self.model.appendRow(QStandardItem('【補足となる文章】'))
        #補足する文章
        #neo4jによる検索
        self.hosoku_bunsyo_net_list=[]
        self.hosoku_zyo_net_list=[]
        self.hosoku_syo_net_list=[]
        self.hosoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`補足`]-(m:sentence) RETURN m.`文書id`;");
        self.hosoku_id_list=[]
        for i in self.hosoku_bunsyo:
            i=str(i)
            self.hosoku_id=re.findall('=(.*)>',i)
            self.hosoku_id_list.append(self.hosoku_id)
        #print(self.hosoku_id_list)
        for p in self.hosoku_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.hosoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.hosoku_bunsyo:
                i=str(i).split("'")
                self.hosoku_bunsyo_result=0
                self.hosoku_bunsyo_result=i[1]
                self.hosoku_bunsyo_net_list.append(self.hosoku_bunsyo_result)
            self.hosoku_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.hosoku_zyo:
                i=str(i).split("'")
                self.hosoku_zyo_result=0
                self.hosoku_zyo_result=i[1]
                self.hosoku_zyo_net_list.append(self.hosoku_zyo_result)
            self.hosoku_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.hosoku_syo:
                i=str(i).split("'")
                self.hosoku_syo_result=0
                self.hosoku_syo_result=i[1]
                self.hosoku_syo_net_list.append(self.hosoku_syo_result)
        #print(self.hosoku_id_list)
        #print(self.hosoku_bunsyo_net_list)
        #print(self.hosoku_zyo_net_list)
        #print(self.hosoku_syo_net_list)

        self.hosoku_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.hosoku_id_list)):
            self.hosoku_item = QStandardItem("ID:"+str(str(self.hosoku_id_list[i]).replace("[","").replace("]","").replace("'",""))+"　タイトル:"+str(self.hosoku_bunsyo_net_list[i])+"　条名:"+str(self.hosoku_zyo_net_list[i])+"　章名:"+str(self.hosoku_syo_net_list[i])+" ")
            self.hosoku_item.setCheckable(True)
            self.hosoku_item.setSelectable(False)
            self.hosoku_item.setCheckState(Qt.Unchecked)
            self.hosoku_item_list.append([self.hosoku_id_list[i],self.hosoku_bunsyo_net_list[i],self.hosoku_zyo_net_list[i],self.hosoku_syo_net_list[i]])
            #print(self.hosoku_item_list)
            self.model.appendRow(self.hosoku_item)


        self.model.appendRow(QStandardItem('【所属となる文章】'))
        #所属文書の検索
        self.syozoku_bunsyo_net_list=[]
        self.syozoku_zyo_net_list=[]
        self.syozoku_syo_net_list=[]
        self.syozoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`所属`]-(m:sentence) RETURN m.`文書id`;");
        self.syozoku_id_list=[]
        for i in self.syozoku_bunsyo:
            i=str(i)
            self.syozoku_id=re.findall('=(.*)>',i)
            self.syozoku_id_list.append(self.syozoku_id)
        #print(self.syozoku_id_list)
        for p in self.syozoku_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            self.syozoku_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.syozoku_bunsyo:
                i=str(i).split("'")
                self.syozoku_bunsyo_result=0
                self.syozoku_bunsyo_result=i[1]
                self.syozoku_bunsyo_net_list.append(self.syozoku_bunsyo_result)
            self.syozoku_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.syozoku_zyo:
                i=str(i).split("'")
                self.syozoku_zyo_result=0
                self.syozoku_zyo_result=i[1]
                self.syozoku_zyo_net_list.append(self.syozoku_zyo_result)
            self.syozoku_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.syozoku_syo:
                i=str(i).split("'")
                self.syozoku_syo_result=0
                self.syozoku_syo_result=i[1]
                self.syozoku_syo_net_list.append(self.syozoku_syo_result)
        #print(self.syozoku_id_list)
        #print(self.syozoku_bunsyo_net_list)
        #print(self.syozoku_zyo_net_list)
        #print(self.syozoku_syo_net_list)

        self.syozoku_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.syozoku_id_list)):
            self.syozoku_item = QStandardItem("ID:"+str(str(self.syozoku_id_list[i]).replace("[","").replace("]","").replace("'",""))+"　タイトル:"+str(self.syozoku_bunsyo_net_list[i])+"　条名:"+str(self.syozoku_zyo_net_list[i])+"　章名:"+str(self.syozoku_syo_net_list[i])+" ")
            self.syozoku_item.setCheckable(True)
            self.syozoku_item.setSelectable(False)
            self.syozoku_item.setCheckState(Qt.Unchecked)
            self.syozoku_item_list.append([self.syozoku_id_list[i],self.syozoku_bunsyo_net_list[i],self.syozoku_zyo_net_list[i],self.syozoku_syo_net_list[i]])
            #print(self.syozoku_item_list)
            self.model.appendRow(self.syozoku_item)

        print("ここから類似文章")
        self.model.appendRow(QStandardItem('【類似度の高い文章】'))
        #類似文章
        #neo4jによる検索
        self.ruizi_bunsyo_net_list=[]
        self.ruizi_zyo_net_list=[]
        self.ruizi_syo_net_list=[]
        self.ruizi_do_net_list =[]
        self.ruizi_bunsyo=session.run("MATCH (n:sentence{文書id:"+str(self.research_id)+"})-[r:`類似`]-(m:sentence) RETURN m.`文書id`;");
        self.ruizi_id_list=[]
        for i in self.ruizi_bunsyo:
            i=str(i)
            self.ruizi_id=re.findall('=(.*)>',i)
            self.ruizi_id_list.append(self.ruizi_id)
        #print(self.ruizi_id_list)
        for p in self.ruizi_id_list:
            p=str(p).replace("[","").replace("]","").replace("'","")
            #print(p)
            self.ruizi_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.ruizi_bunsyo:
                i=str(i).split("'")
                self.ruizi_bunsyo_result=0
                self.ruizi_bunsyo_result=i[1]
                self.ruizi_bunsyo_net_list.append(self.ruizi_bunsyo_result)
            self.ruizi_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.ruizi_zyo:
                i=str(i).split("'")
                self.ruizi_zyo_result=0
                self.ruizi_zyo_result=i[1]
                self.ruizi_zyo_net_list.append(self.ruizi_zyo_result)
            self.ruizi_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.ruizi_syo:
                i=str(i).split("'")
                self.ruizi_syo_result=0
                self.ruizi_syo_result=i[1]
                self.ruizi_syo_net_list.append(self.ruizi_syo_result)
            self.ruizi_do=session.run("MATCH (n:sentence{文書id:"+ str(p)+"})-[r:`類似`]-(m:sentence{文書id:"+str(self.research_id)+"}) RETURN r.類似度;");
            for i in self.ruizi_do:
                i=str(i)
                self.ruizi_do_result=0
                self.ruizi_do_result=re.findall('=(.*)>',i)
                self.ruizi_do_net_list.append(self.ruizi_do_result)
        #print(self.ruizi_id_list)
        #print(self.ruizi_bunsyo_net_list)
        #print(self.ruizi_zyo_net_list)
        #print(self.ruizi_syo_net_list)
        #print(self.ruizi_do_net_list)
        self.ruizi_item_list=[]
        #チェックボタンの判断
        for i in range(len(self.ruizi_id_list)):
            self.ruizi_item = QStandardItem("ID:"+str(str(self.ruizi_id_list[i]).replace("[","").replace("]","").replace("'",""))+"　タイトル:"+str(self.ruizi_bunsyo_net_list[i])+"　条名:"+str(self.ruizi_zyo_net_list[i])+"　章名:"+str(self.ruizi_syo_net_list[i])+"　類似度:"+str(str(self.ruizi_do_net_list[i]).replace("[","").replace("]","").replace("'",""))+" ")
            self.ruizi_item.setCheckable(True)
            self.ruizi_item.setSelectable(False)
            self.ruizi_item.setCheckState(Qt.Unchecked)
            self.ruizi_item_list.append([self.ruizi_id_list[i],self.ruizi_bunsyo_net_list[i],self.ruizi_zyo_net_list[i],self.ruizi_syo_net_list[i]])
            #print(self.ruizi_item_list)
            self.model.appendRow(self.ruizi_item)

        print("ここから保存文章")
        #チェックボックスに入れられた文章を保存
        self.model.appendRow(QStandardItem('【保存文章】'))
        #チェックボックスの判断
        print(self.hozon_list)
        print(self.check_list)
        self.hozon_list_new =[]
        self.check_check_list=[]
        for i in range(len(self.hozon_list)):
            if i in self.check_list:
                print(i)
                self.check_check_list.append(i)
                self.check_number=0
                for l in self.check_check_list:
                    if self.hozon_list[i][0]==self.hozon_list[l][0]:
                        self.check_number=self.check_number+1
                if self.check_number==1:
                    self.hozon_item = QStandardItem("ID:"+str(str(self.hozon_list[i][0]).replace("[","").replace("]","").replace("'",""))+"　タイトル:"+str(self.hozon_list[i][1])+" 条名:"+str(self.hozon_list[i][2])+"　章名:"+str(self.hozon_list[i][2])+" ")
                    self.hozon_item.setCheckable(True)
                    self.hozon_item.setSelectable(False)
                    self.hozon_item.setCheckState(Qt.Checked)
                    self.hozon_list_new.append([self.hozon_list[i][0],self.hozon_list[i][1],self.hozon_list[i][2],self.hozon_list[i][3]])
                    self.model.appendRow(self.hozon_item)

        self.SubgroupBox.setModel(self.model)
        self.SubgroupBox.setLayout(self.SubformLayout)
        self.Subscroll=QtWidgets.QScrollArea()
        self.Subscroll.setWidget(self.SubgroupBox)
        self.Subscroll.setWidgetResizable(True)
        self.Subscroll.setFixedHeight(500)
        self.grid.addWidget(self.Subscroll,10,0)
        self.model.appendRow(QStandardItem(''))
        self.model.appendRow(QStandardItem(''))

# SubWindowを表示する
    def makeWindow(self):
        subWindow = SubWindow(self)
        subWindow.show()

class StandardItem(QStandardItem):
    def __init__(self, txt=' ', font_size=3, set_bold=False, color=QColor(255,255,255)):
        super().__init__()
        fnt = QFont('Open Sans',font_size)
        fnt.setBold(set_bold)
        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)

class SubWindow:
    def __init__(self, MainWindow:MainWindow):
        self.w = QtWidgets.QDialog()
        self.vertical = QtWidgets.QVBoxLayout()
        self.treeview = QtWidgets.QTreeView()
        self.treeview.setGeometry(0, 0,1500,750)
        self.treeview.setColumnWidth(110,70)
        self.treemodel = QStandardItemModel()
        self.treeview.setModel(self.treemodel)
        self.treeview.expandAll()
        self.treeview.doubleClicked.connect(self.get)
        self.vertical.addWidget(self.treeview)
        # 対象データ
        self.df_candid_vector = MainWindow.df_candid_vector
        self.kouho_item_list= MainWindow.kouho_item_list2
        self.konkyo_item_list= MainWindow.konkyo_item_list2
        self.hosoku_item_list= MainWindow.hosoku_item_list2
        self.syozoku_item_list= MainWindow.syozoku_item_list2
        self.ruizi_item_list= MainWindow.ruizi_item_list2
        self.hozon_list_new= MainWindow.hozon_list_new2
        self.check_list=MainWindow.check_list2

        self.w.setGeometry(0, 0, 1500,750)
        self.w.setLayout(self.vertical)

        ################################
        #表示システムの変更
        self.output_result_subwindow()
        ###############################
    # ここで親ウィンドウに値を渡している
    def setParamOriginal(self):
        self.parent.setParam(self.edit.text())

    def show(self):
        self.w.exec_()

    def get(self):
        print('成功')

    def output_result_subwindow(self):
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=basic_auth("neo4j", "handhara66"))
        session = driver.session()
        rootNode = self.treemodel.invisibleRootItem()
        #top_num = 9
        #self.df_candid_vector.to_csv('../../data/search_textdata_temp/結果.csv', encoding='utf_8_sig')
        #self.df_candid_vector_descending = self.df_candid_vector[self.df_candid_vector.result_dendro==(self.max_sim_cluster_num+1)].sort_values('sim', ascending=False)[:50]

        self.hozon_list=[]
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.kouho_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.konkyo_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.hosoku_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.syozoku_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.ruizi_item_list)
        self.hozon_list.extend([["0","0","0","0"]])
        self.hozon_list.extend(self.hozon_list_new)

        #idリストにチェックボックスの文章を格納
        self.hyouzi_id_list=[]

        for i in range(len(self.hozon_list)):
            if i in self.check_list:
                print(i)
                self.hyouzi_id_list.append(str(str(self.hozon_list[i][0]).replace("[","").replace("]","").replace("'","")))
        print(self.hyouzi_id_list)

        #neo4jによる呼び込み
        self.text_name_list=[]
        self.hen_name_list=[]
        self.syo_name_list=[]
        self.setu_name_list=[]
        self.ko_name_list=[]
        self.sentence_name_list=[]
        self.zyo_name_list=[]

        for p in self.hyouzi_id_list:
            #print(p)
            self.hyouzi_bunsyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文書名;");
            for i in self.hyouzi_bunsyo:
                i=str(i).split("'")
                #print(i)
                self.text_name_list.append(i[1])
            #print(self.text_name_list)
            self.hyouzi_zyo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.条;");
            for i in self.hyouzi_zyo:
                i=str(i).split("'")
                self.zyo_name_list.append(i[1])
            self.hyouzi_syo=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.章;");
            for i in self.hyouzi_syo:
                i=str(i).split("'")
                self.syo_name_list.append(i[1])
            self.hyouzi_hen=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.編;");
            for i in self.hyouzi_hen:
                i=str(i).split("'")
                self.hen_name_list.append(i[1])
            self.hyouzi_ko=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.項;");
            for i in self.hyouzi_ko:
                i=str(i).split("'")
                self.ko_name_list.append(i[1])
            self.hyouzi_setu=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.節;");
            for i in self.hyouzi_setu:
                i=str(i).split("'")
                self.setu_name_list.append(i[1])
            self.hyouzi_sentence=session.run("MATCH (n:sentence{文書id:"+ str(p)+ "}) RETURN n.文章;");
            for i in self.hyouzi_sentence:
                i=str(i).split("'")
                self.hyouzi_sentence_result=i[1]
                self.sentence_name_list.append(self.hyouzi_sentence_result)

        #self.p=1
        """
        #if self.p==1:
        self.k=[]
        self.k_index=[]
        for i in self.hyouzi_id_list:

            self.k.extend(self.df_candid_vector["id"])
            self.k_index.append(self.k.index(i))
        print(self.k)
        print(self.k_index)
        self.df_candid_vector_descending=[]
        for i in self.k_index:
            print(self.df_candid_vector[:,i])
            self.df_candid_vector_descending.append(self.df_candid_vector[:,i])
        print(self.df_candid_vector_descending)
        #self.df_candid_vector_descending.append()

        self.df_candid_vector_descending = self.df_candid_vector.sort_values('sim', ascending=False)[:10]
        # 確認用にcsv形式で保存
        self.df_candid_vector_descending.to_csv('../../data/result_temp/結果_降順_確認用.csv', encoding='utf_8_sig')
        # 結果の表示データ　ArticleTitle
        self.text_name_list=self.df_candid_vector_descending['文書の名称'].tolist()
        self.hen_name_list=self.df_candid_vector_descending['編'].tolist()
        self.hen_name_name_list=self.df_candid_vector_descending['編の名称'].tolist()
        self.syo_name_list=self.df_candid_vector_descending['章'].tolist()
        self.syo_name_name_list=self.df_candid_vector_descending['章の名称'].tolist()
        self.setu_name_list=self.df_candid_vector_descending['節'].tolist()
        self.setu_name_name_list=self.df_candid_vector_descending['節の名称'].tolist()
        self.ko_name_list=self.df_candid_vector_descending['項'].tolist()
        self.ko_name_name_list=self.df_candid_vector_descending['項の名称'].tolist()
        self.sentence_name_list=self.df_candid_vector_descending['Sentence'].tolist()
        self.zyo_name_list=self.df_candid_vector_descending['条'].tolist()
        self.ArticleTitle_name_list=self.df_candid_vector_descending['ArticleTitle_clensed'].tolist()
        self.sansyo_name_list=self.df_candid_vector_descending['参照先名'].tolist()
        self.cos_list = self.df_candid_vector_descending['sim'].tolist()
        """

        for num_text in range(len(self.hyouzi_id_list)):
            self.lbl = StandardItem('【候補文章 ID:'+str(self.hyouzi_id_list[num_text])+'】', 20, set_bold=True)
            if str(self.text_name_list[num_text]) != 'nan':
                self.text = StandardItem("文書名:"+str(self.text_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.text)
            if str(self.hen_name_list[num_text]) != 'nan':
                self.hen = StandardItem("編:"+str(self.hen_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.hen)
            if str(self.syo_name_list[num_text]) != 'nan':
                self.syo = StandardItem("章:"+str(self.syo_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.syo)
            if str(self.setu_name_list[num_text]) != 'nan':
                self.setu = StandardItem("節:"+str(self.setu_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.setu)
            if str(self.ko_name_list[num_text]) != 'nan':
                self.ko = StandardItem("項:"+str(self.ko_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.ko)
            if str(self.zyo_name_list[num_text]) != 'nan':
                self.ko = StandardItem("条:"+str(self.zyo_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.ko)
            self.naiyou = StandardItem("内容", 16, set_bold=True)
            self.lbl.appendRow(self.naiyou)
            if len(str(self.sentence_name_list[num_text])) <= 100:
                self.sentence = StandardItem(str(self.sentence_name_list[num_text]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence)

            elif len(str(self.sentence_name_list[num_text])) <= 200:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)
            elif len(str(self.sentence_name_list[num_text])) <= 300:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)
            elif len(str(self.sentence_name_list[num_text])) <= 400:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)
            elif len(str(self.sentence_name_list[num_text])) <= 500:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)
            elif len(str(self.sentence_name_list[num_text])) <= 600:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)
            elif len(str(self.sentence_name_list[num_text])) <= 700:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)
            elif len(str(self.sentence_name_list[num_text])) <= 800:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)
            elif len(str(self.sentence_name_list[num_text])) <= 900:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)
            elif len(str(self.sentence_name_list[num_text])) <= 1000:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)
            elif len(str(self.sentence_name_list[num_text])) <= 1100:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)
            elif len(str(self.sentence_name_list[num_text])) <= 1200:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)
            elif len(str(self.sentence_name_list[num_text])) <= 1300:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)
            elif len(str(self.sentence_name_list[num_text])) <= 1400:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)
            elif len(str(self.sentence_name_list[num_text])) <= 1500:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:1400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)

                self.sentence15 = StandardItem(str(self.sentence_name_list[num_text][1400:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence15)
            elif len(str(self.sentence_name_list[num_text])) <= 1600:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:1400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)

                self.sentence15 = StandardItem(str(self.sentence_name_list[num_text][1400:1500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence15)

                self.sentence16 = StandardItem(str(self.sentence_name_list[num_text][1500:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence16)
            else:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:1400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)

                self.sentence15 = StandardItem(str(self.sentence_name_list[num_text][1400:1500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence15)

                self.sentence16 = StandardItem(str(self.sentence_name_list[num_text][1500:1600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence16)

                self.sentence17 = StandardItem(str(self.sentence_name_list[num_text][1600:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence17)

            #if str(self.sansyo_name_list[num_text]) != 'nan':

                if len(str(self.sentence_name_list[num_text])) <= 100:
                    self.sansyo1 = StandardItem(str(self.sansyo_name_list[num_text]), 14, set_bold=True)
                    self.lbl.appendRow(self.sansyo1)
                elif len(str(self.sentence_name_list[num_text])) <= 200:
                    self.sansyo1 = StandardItem(str(self.sansyo_name_list[num_text][:100]), 14, set_bold=True)
                    self.lbl.appendRow(self.sansyo1)

                    self.sansyo2 = StandardItem(str(self.sansyo_name_list[num_text][100:]), 14, set_bold=True)
                    self.lbl.appendRow(self.sansyo2)
            rootNode.appendRow(self.lbl)

    def output_result_subwindow2(self):
        rootNode = self.treemodel.invisibleRootItem()
        top_num = 15
        #self.df_candid_vector.to_csv('../../data/search_textdata_temp/結果.csv', encoding='utf_8_sig')
        #self.df_candid_vector_descending = self.df_candid_vector[self.df_candid_vector.result_dendro==(self.max_sim_cluster_num+1)].sort_values('sim', ascending=False)[:50]
        self.df_candid_vector_descending = self.df_candid_vector.sort_values('sim', ascending=False)[:10]
        # 確認用にcsv形式で保存
        self.df_candid_vector_descending.to_csv('../../data/result_temp/結果_降順_確認用.csv', encoding='utf_8_sig')

        # 結果の表示データ　ArticleTitle
        self.text_name_list=self.df_candid_vector_descending['文書の名称'].tolist()
        self.hen_name_list=self.df_candid_vector_descending['編'].tolist()
        self.hen_name_name_list=self.df_candid_vector_descending['編の名称'].tolist()
        self.syo_name_list=self.df_candid_vector_descending['章'].tolist()
        self.syo_name_name_list=self.df_candid_vector_descending['章の名称'].tolist()
        self.setu_name_list=self.df_candid_vector_descending['節'].tolist()
        self.setu_name_name_list=self.df_candid_vector_descending['節の名称'].tolist()
        self.ko_name_list=self.df_candid_vector_descending['項'].tolist()
        self.ko_name_name_list=self.df_candid_vector_descending['項の名称'].tolist()
        self.sentence_name_list=self.df_candid_vector_descending['Sentence'].tolist()
        self.zyo_name_list=self.df_candid_vector_descending['条'].tolist()
        self.ArticleTitle_name_list=self.df_candid_vector_descending['ArticleTitle_clensed'].tolist()
        self.sansyo_name_list=self.df_candid_vector_descending['参照先名'].tolist()
        self.cos_list = self.df_candid_vector_descending['sim'].tolist()

        for num_text in range(top_num):
            self.lbl = StandardItem('【'+str(num_text+1)+'番目の候補】', 20, set_bold=True)
            if str(self.text_name_list[num_text]) != 'nan':
                self.text = StandardItem(str(self.text_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.text)
            if str(self.hen_name_list[num_text]) != 'nan':
                self.hen = StandardItem(str(self.hen_name_list[num_text])+str(self.hen_name_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.hen)
            if str(self.syo_name_list[num_text]) != 'nan':
                self.syo = StandardItem(str(self.syo_name_list[num_text])+str(self.syo_name_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.syo)
            if str(self.setu_name_list[num_text]) != 'nan':
                self.setu = StandardItem(str(self.setu_name_list[num_text])+str(self.setu_name_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.setu)
            if str(self.ko_name_list[num_text]) != 'nan':
                self.ko = StandardItem(str(self.ko_name_list[num_text])+str(self.ko_name_name_list[num_text]), 16, set_bold=True)
                self.lbl.appendRow(self.ko)

            if len(str(self.sentence_name_list[num_text])) <= 100:
                self.sentence = StandardItem(str(self.sentence_name_list[num_text]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence)

            elif len(str(self.sentence_name_list[num_text])) <= 200:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)
            elif len(str(self.sentence_name_list[num_text])) <= 300:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)
            elif len(str(self.sentence_name_list[num_text])) <= 400:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)
            elif len(str(self.sentence_name_list[num_text])) <= 500:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)
            elif len(str(self.sentence_name_list[num_text])) <= 600:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)
            elif len(str(self.sentence_name_list[num_text])) <= 700:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)
            elif len(str(self.sentence_name_list[num_text])) <= 800:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)
            elif len(str(self.sentence_name_list[num_text])) <= 900:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)
            elif len(str(self.sentence_name_list[num_text])) <= 1000:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)
            elif len(str(self.sentence_name_list[num_text])) <= 1100:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)
            elif len(str(self.sentence_name_list[num_text])) <= 1200:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)
            elif len(str(self.sentence_name_list[num_text])) <= 1300:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)
            elif len(str(self.sentence_name_list[num_text])) <= 1400:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)
            elif len(str(self.sentence_name_list[num_text])) <= 1500:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:1400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)

                self.sentence15 = StandardItem(str(self.sentence_name_list[num_text][1400:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence15)
            elif len(str(self.sentence_name_list[num_text])) <= 1600:

                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:1400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)

                self.sentence15 = StandardItem(str(self.sentence_name_list[num_text][1400:1500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence15)

                self.sentence16 = StandardItem(str(self.sentence_name_list[num_text][1500:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence16)
            else:
                self.sentence1 = StandardItem(str(self.sentence_name_list[num_text][:100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence1)

                self.sentence2 = StandardItem(str(self.sentence_name_list[num_text][100:200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence2)

                self.sentence3 = StandardItem(str(self.sentence_name_list[num_text][200:300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence3)

                self.sentence4 = StandardItem(str(self.sentence_name_list[num_text][300:400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence4)

                self.sentence5 = StandardItem(str(self.sentence_name_list[num_text][400:500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence5)

                self.sentence6 = StandardItem(str(self.sentence_name_list[num_text][500:600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence6)

                self.sentence7 = StandardItem(str(self.sentence_name_list[num_text][600:700]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence7)

                self.sentence8 = StandardItem(str(self.sentence_name_list[num_text][700:800]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence8)

                self.sentence9 = StandardItem(str(self.sentence_name_list[num_text][800:900]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence9)

                self.sentence10 = StandardItem(str(self.sentence_name_list[num_text][900:1000]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence10)

                self.sentence11 = StandardItem(str(self.sentence_name_list[num_text][1000:1100]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence11)

                self.sentence12 = StandardItem(str(self.sentence_name_list[num_text][1100:1200]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence12)

                self.sentence13 = StandardItem(str(self.sentence_name_list[num_text][1200:1300]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence13)

                self.sentence14 = StandardItem(str(self.sentence_name_list[num_text][1300:1400]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence14)

                self.sentence15 = StandardItem(str(self.sentence_name_list[num_text][1400:1500]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence15)

                self.sentence16 = StandardItem(str(self.sentence_name_list[num_text][1500:1600]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence16)

                self.sentence17 = StandardItem(str(self.sentence_name_list[num_text][1600:]), 14, set_bold=True)
                self.lbl.appendRow(self.sentence17)

            if str(self.sansyo_name_list[num_text]) != 'nan':

                if len(str(self.sentence_name_list[num_text])) <= 100:
                    self.sansyo1 = StandardItem(str(self.sansyo_name_list[num_text]), 14, set_bold=True)
                    self.lbl.appendRow(self.sansyo1)
                elif len(str(self.sentence_name_list[num_text])) <= 200:
                    self.sansyo1 = StandardItem(str(self.sansyo_name_list[num_text][:100]), 14, set_bold=True)
                    self.lbl.appendRow(self.sansyo1)

                    self.sansyo2 = StandardItem(str(self.sansyo_name_list[num_text][100:]), 14, set_bold=True)
                    self.lbl.appendRow(self.sansyo2)
            rootNode.appendRow(self.lbl)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
