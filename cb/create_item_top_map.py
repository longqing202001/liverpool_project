import re
import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy import sparse
import sklearn.preprocessing as pp
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

def similarity_func(u, v):
    return 1/(1+euclidean(u,v))

def topN(df,indexid,n):
  ranked=[]
  item_scores=df.loc[indexid,:].tolist()
  sorted_list = sorted( [(x,i) for (i,x) in enumerate(item_scores)], reverse=True )[:n]
  for item in sorted_list:
    items=[]
    items.append(item_dic[item[1]])
    items.append(item[0])
    ranked.append(items)
  return ranked

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
  df=df[df["itemid"].isin(indexes)]
  #print(len(df))
  #split_mask= df['itemid'] in indexes
  #trainTR = df[split_mask]
  #df[df['itemid'].isin(indexes)]
  #df.index = df.index.astype(int)
  #df.reindex(indexes, level=0)
  #df[df.index.levels[0].isin(indexes)]
  #df[df.index.map(lambda x: x[0] in indexes)]
  #mask=(df.index in indexes)
  #indexes=[30992]
  #df.loc[indexes,:]
  df.to_csv(newfilepath, encoding='utf-8',index=False)

# read dic info from file
preprocessed_data_path="/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/"
preprocessed_data_path="/home/erlangshen2019/project/"
indexes_dic_path=preprocessed_data_path+"item_dic.txt"
item_dic=readItemDic(indexes_dic_path)

#filter rows by item indexes
item_features_path=preprocessed_data_path+"item_features.csv"
#new_item_features_path=preprocessed_data_path+"item_features_filter.csv"
#indexes = list(map(int, list(item_dic.values())))
#filterItems(item_features_path,new_item_features_path,indexes)

DF_var=pd.read_csv(item_features_path)
DF_var.index = DF_var["itemid"]
#print(DF_var[0:10])
DF_var = DF_var.drop('itemid', 1)
#print(DF_var[0:10])
#DF_var.drop('itemid', axis=1)
#DF_var.drop('itemid', axis=1)
#DF_var["itemid"]=DF_var["itemid"].astype(int)
#DF_var.index = DF_var["itemid"]
#DF_var.drop('itemid', axis=1)
print(DF_var.values)

#item_matrix_path=preprocessed_data_path+"item_matrix.csv"
#A_sparse = sparse.csr_matrix(DF_var)
#A_sparse=DF_var.values

cosine_sim=cosine_similarity(DF_var,DF_var)
print(np.shape(cosine_sim))

topk=100+1
#score_matrix=np.array(cosine_sim)
topk_ind = cosine_sim.argsort()[:,::-1][:,:topk]
print(np.shape(topk_ind))

index_map = {i:v for i,v in enumerate(DF_var.index)}
topk_dic=dict()
for zipitem in zip(DF_var.index,topk_ind):
  top_k=[]
  for i in zipitem[1]:
    temp_id=index_map[i]
    if temp_id!=zipitem[0]:
      top_k.append(temp_id)
  topk_dic[zipitem[0]]=top_k
print(len(topk_dic.keys()))
print("item topk lenghts:"+str(len(topk_dic)))
target_filepath=preprocessed_data_path+"item_topk.dat"
with open(target_filepath, "wb") as f:
  pickle.dump(topk_dic, f)
