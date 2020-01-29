import pandas as pd
import numpy as np
import re
import math
from keras.layers.embeddings import Embedding

#base_path="/home/ryu/Desktop/project/liverpoor/"
#preprocessed_data_path=base_path+"data/preprocessed_data/"

def readItemDic(item_dic_file):
        new_dict = dict()
        with open(item_dic_file) as fp:
            for cnt, line in enumerate(fp):
                temp=re.split(r' ', line.strip('\n'))
                new_dict[int(temp[1])]=int(temp[0])
                #print("Line {}: {}".format(cnt, line))
        return new_dict

def getEmbeddingLayer(item_features_path,item_dic_path):
        #feature_num=132
        item_dic,indexes=getItemDicInfo(item_dic_path)
        num_items=len(item_dic)
        #item_features_path=preprocessed_data_path+"item_features.csv"
        df=pd.read_csv(item_features_path)
        #df=pd.read_csv(item_features_path,index_col=0)
        #df=df[df.itemid.isin(indexes)]
        df=df[df["itemid"].isin(indexes)]
        ### create item-vector dictionary
        item_vector = {}
        for row in df.values.tolist():
            item_vector[int(row[0])]=row[1:]
        feature_num=len(df.columns)-1
        embedding_matrix = np.zeros((num_items, feature_num), dtype=np.float32)
        for i in item_dic.keys():
            #print(itemid_map_vector[i])
            if item_dic[i] in item_vector:
                embedding_matrix[i] = np.array(item_vector[item_dic[i]][:feature_num])
        feature_embedding_layer = Embedding(num_items,
                            feature_num,
                            weights=[embedding_matrix],
                            name = 'feature_embedding_vector',
                            trainable=False)
        return embedding_matrix,feature_embedding_layer


def getItemDicInfo(item_dic_path):
        #item_dic_path=preprocessed_data_path+"item_dic.txt"
        item_dic=readItemDic(item_dic_path)
        num_items=len(item_dic)
        indexes = list(map(int, list(item_dic.values())))
        return item_dic,indexes


def getHitRatio(ranklist, gtItem):
        if gtItem in ranklist:
            return 1
        return 0

def getNDCG(ranklist, gtItem):
        ar = np.array(ranklist)
        if gtItem in ar:
            return math.log(2) / math.log(np.where(ar == gtItem)[0][0] + 2)
        return 0