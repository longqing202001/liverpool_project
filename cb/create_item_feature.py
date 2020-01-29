from sklearn import preprocessing
import numpy as np 
import pandas as pd
import time
import pickle
import os
import csv
import re
import category_encoders as ce
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import decomposition

def readItemDic(item_dic_file):
  new_dict = dict()
  with open(item_dic_file) as fp:
    for cnt, line in enumerate(fp):
      temp=re.split(r' ', line.strip('\n'))
      new_dict[int(temp[1])]=int(temp[0])
      #print("Line {}: {}".format(cnt, line))
  return new_dict

def filterItems(rawfilepath,newfilepath,indexes):
  df=pd.read_csv(rawfilepath,index_col=0)
  df=df[df.itemid.isin(indexes)]
  df.to_csv(newfilepath, encoding='utf-8',index=False)

preprocessed_data_path="/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/"
attribute_path=preprocessed_data_path+'item_attribute.csv'

indexes_dic_path=preprocessed_data_path+"item_dic.txt"
item_dic=readItemDic(indexes_dic_path)
indexes = list(map(int, list(item_dic.values())))

new_attribute_path=preprocessed_data_path+"item_attribute_filter.csv"
filterItems(attribute_path,new_attribute_path,indexes)

# normallize colomn "790"
df=pd.read_csv(new_attribute_path)
texts = df["283"]+ " " + df["888"]
index_ids = df[['itemid']].values.astype(int)
texts=texts.values.tolist()
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts) 
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

print("Computing PCA projection")
X_pca = decomposition.TruncatedSVD(n_components=500).fit_transform(tfidf_matrix)
print(X_pca.shape)

x_features = pd.DataFrame(X_pca)
x_itemids = pd.DataFrame(index_ids,columns=['itemid'])

new_df = pd.concat([x_itemids, x_features], axis=1, sort=False)
print(new_df.shape)
item_features_path=preprocessed_data_path+'item_features.csv'
new_df.to_csv(item_features_path, encoding='utf-8',index=False)


