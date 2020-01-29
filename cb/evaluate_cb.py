import time
import numpy as np
import pickle
import pandas as pd
import re
import math
from time import time

def readItemDic(item_dic_file):
  new_dict = dict()
  with open(item_dic_file) as fp:
    for cnt, line in enumerate(fp):
      temp=re.split(r' ', line.strip('\n'))
      new_dict[int(temp[1])]=int(temp[0])
      #print("Line {}: {}".format(cnt, line))
  return new_dict

def topN(df,indexid,n):
  item_scores=df.loc[indexid,:].tolist()
  sorted_list = sorted( [(x,i) for (i,x) in enumerate(item_scores)], reverse=True )[:n]
  ranklist=[]
  scores=[]
  for item in sorted_list:
    ranklist.append(item_dic[item[1]])
    scores.append(item[0])
  return ranklist,scores

def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0

def getNDCG(ranklist, gtItem):
    ar = np.array(ranklist)
    if gtItem in ar:
        return math.log(2) / math.log(np.where(ar == gtItem)[0][0] + 2)
    return 0

preprocessed_data_path="/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/"
topK=50

indexes_dic_path=preprocessed_data_path+"item_dic.txt"
item_dic=readItemDic(indexes_dic_path)

item_top_dic_path=preprocessed_data_path+"item_topk.dat"
with open(item_top_dic_path, 'rb') as f:
    item_top_dic = pickle.load(f)
print(len(item_top_dic.keys()))

inputfilepath=preprocessed_data_path+"valid_input.dat"
with open(inputfilepath, 'rb') as f:
    inputlist = pickle.load(f)

targetfilepath=preprocessed_data_path+"valid_target.dat"
with open(targetfilepath, 'rb') as f:
    targetlist = pickle.load(f)

print(len(inputlist))
print(len(targetlist))

time_1 = time()
hits=[]
ndcgs=[]
i=0
for item in zip(inputlist,targetlist):
    itemid=item_dic[item[0][1]]
    #print("itemid:"+str(itemid))
    targetid=item_dic[item[1]]
    if itemid in item_top_dic:
      ranklist=item_top_dic[itemid]   
      ranklist=ranklist[0:topK]
      #print("rank list:"+str(ranklist))      
      hr = getHitRatio(ranklist, targetid)
      ndcg = getNDCG(ranklist, targetid)
      hits.append(hr)
      ndcgs.append(ndcg)
      i+=1
mean_hr, mean_ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('HR = %.4f, NDCG = %.4f [%.1f s] ,evaluated: %d' % (mean_hr, mean_ndcg, time()-time_1,i))