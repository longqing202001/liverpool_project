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

#preprocessed_data_path="/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/"
preprocessed_data_path="/home/erlangshen2019/project/"
attribute_path=preprocessed_data_path+'item_attribute.csv'

indexes_dic_path=preprocessed_data_path+"item_dic.txt"
item_dic=readItemDic(indexes_dic_path)
indexes = list(map(int, list(item_dic.values())))

new_attribute_path=preprocessed_data_path+"item_attribute_filter.csv"
filterItems(attribute_path,new_attribute_path,indexes)

# normallize colomn "790"
df=pd.read_csv(new_attribute_path)
df=df.drop('available', axis=1)
df['text'] = df["283"]+ " " + df["888"]
#df['text'] = df[['283', '888', '678', '917', '202', '6', '839', '227', '698', '689']].apply(lambda x: ' '.join(x), axis=1)
#df['text'] = df[['283', '888']].apply(lambda x: ' '.join(x), axis=1)
#df['text'] = df['283']+df['888']+df['678']+df['917']+df['202'] + df['6']+df['839']+df['227'] +df['698'] +df['689'] +df['28'] + df['928']
df=df.drop('112', axis=1)
df=df.drop('764', axis=1)
df=df.drop('159', axis=1)
df["790"] = df["790"].str.replace("n", "")

df=df.drop('283', axis=1)
df=df.drop('888', axis=1)
df=df.drop('678', axis=1)
df=df.drop('917', axis=1)
df=df.drop('202', axis=1)
df=df.drop('6', axis=1)
df=df.drop('839', axis=1)
df=df.drop('227', axis=1)
df=df.drop('698', axis=1)
df=df.drop('689', axis=1)
df=df.drop('28', axis=1)
df=df.drop('928', axis=1)

filter_attribute_path=preprocessed_data_path+'item_attribute_filter.csv'
df.to_csv(filter_attribute_path,index=False)

# normallize colomn "790"
x = df[['790']].values.astype(float)
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()
# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)
# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)
df_normalized.columns = ["790"]
# update column "790"
df.update(df_normalized)

# Use binary encoding to encode categorical variables
encoder = ce.BinaryEncoder(cols=['categoryid','364','776']).fit(df)
# Convert data
df = encoder.transform(df) 
np.set_printoptions(precision=2)
col_283=df['text'].to_numpy()
vectorizer = TfidfVectorizer()
#vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b',min_df=0.03, max_df=0.8)
#vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(col_283)
tiidf_df = pd.DataFrame(data=vecs.toarray())
print("shape: ")
print(tiidf_df.shape)
#print("vocabulary: ")
#print(vectorizer.vocabulary_)
#print(vectorizer.get_feature_names())

df=df.drop('text', axis=1)
df = pd.concat([df, tiidf_df], axis=1, sort=False)
# index id
index_ids = df[['itemid']].values.astype(int)
features_df=df.drop('itemid', axis=1)
print("features shape: ")
print(features_df.shape)

item_features_path=preprocessed_data_path+'item_features.csv'
df.to_csv(item_features_path, encoding='utf-8',index=False)