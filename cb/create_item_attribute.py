import numpy as np 
import pandas as pd
import seaborn as sns
import operator
import time
import matplotlib.pyplot as plt
import os
from pathlib import Path
import csv
import category_encoders as ce
import datetime

raw_data_path="/home/ryu/Desktop/project/liverpoor/data/raw_data/"
preprocessed_data_path="/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/"

# concat two file into one
item1_path=raw_data_path+"item_properties_part1.csv"
items1=pd.read_csv(item1_path)
item2_path=raw_data_path+"item_properties_part2.csv"
items2=pd.read_csv(item2_path)
items=pd.concat([items1,items2])
items = items.sort_values(by='timestamp')
print('items shape: ',items.shape)
print(items.shape)
print(items.head())

property_list=items.property.unique()
print("All item properties:"+ str(len(property_list)))
print(property_list[0:10])

itemid_list=items.itemid.unique()
print("All itemid:"+ str(len(itemid_list)))
print(itemid_list[0:10])

items=items.sort_values('timestamp',ascending=False).drop_duplicates(['itemid','property'])
print(items.shape)
print(items.head())

# Count the number of occurrences of each attribute
item_occurrences_list = items.property.value_counts()
item_occurrences_list.sort_index().plot.barh()
#print(item_occurrences_list)

# top 20 properties:
#['available', '364', 'categoryid', '888', '159', '283', '112', '764', '790', '678', '917', '202', '6', '776', '839', '227', '698', '689', '28', '928']
top20_property=item_occurrences_list[0:20]
property_list=top20_property.index.tolist()
print("top 20 properties:")
print(property_list)

items=items[items.property.isin(property_list)]
print(items.shape)
print(items.head())
i=0
item_features=[]
for item_id in itemid_list:
    item_property_data=[]
    item_property_data.append(item_id)
    item_property_list=items[items['itemid']==item_id]
    property_dic={}
    for index, row in item_property_list.iterrows():
        property_dic[row['property']] = row['value']
    for p in property_list:
        is_add=False
        if p in property_dic:
            item_property_data.append(property_dic[p])
            is_add=True
        if not is_add:
            item_property_data.append("")
    if i % 1000==0:
        print(str(i)+" rows produced...")
        print(datetime.datetime.now())
    item_features.append(item_property_data)
    i+=1                                
# iterate over rows with iterrows()
#item_features=[]
#item_dict = {}
#for index, row in items.iterrows():
    #print()
     # access data using column names
     #print(index, row['itemid'], row['property'], row['value'])
fieldnames=['itemid']
for property_name in property_list:
    fieldnames.append(property_name)     
df = pd.DataFrame.from_records(item_features, columns=fieldnames)
attribute_path=preprocessed_data_path+'item_attribute.csv'
df.to_csv(attribute_path)