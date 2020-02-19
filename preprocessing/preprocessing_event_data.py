import numpy as np
import pandas as pd
import datetime
import pickle

def removeShortSessions(data):
    #delete sessions of length < 2
    #group by visitorid and get size 
    sessionLen = data.groupby('visitorid').size() 
    data = data[np.in1d(data.visitorid, sessionLen[sessionLen > 2].index)]
    return data

def removeRareItems(data):
    #delete items of length < 5
    #group by itemid and get size
    itemLen = data.groupby('itemid').size() 
    data = data[np.in1d(data.itemid, itemLen[itemLen > 5].index)]
    return data

#Path to Original interaction csv file
preprocessed_data_path="/home/ryu/Desktop/project/liverpoor/data/raw_data/"
dataBefore = preprocessed_data_path+'events.csv' 
#Path to Processed Folder
dataAfter = "/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/" 

attribute_path=dataAfter+'item_attribute.csv'
df=pd.read_csv(attribute_path)
itemid=df["itemid"].values.tolist()

train = pd.read_csv(dataBefore, sep=',', usecols=['visitorid','timestamp','itemid','event'])
print(len(train))
train= train[train['event'] == 'view']
print(len(train))
train=train[train.itemid.isin(itemid)]
print(len(train))
train.sort_values(["visitorid","timestamp"], inplace=True)
train['match'] = (train.visitorid.eq(train.visitorid.shift()) & train.itemid.eq(train.itemid.shift()))
train = train[~train['match']]


# pre_visitorid=""
# pre_itemid=""
# i=0
# indexes=[]
# print(len(train))
# for index, row in train.iterrows():
#     if row["visitorid"]==pre_visitorid and row["itemid"]==pre_itemid:
#         indexes.append(index)
#         #train.drop(train.index[index])
#     pre_visitorid=row["visitorid"]
#     pre_itemid=row["itemid"]
#     i+=1

# with open('listfile.txt', 'w') as filehandle:
#     for listitem in indexes:
#         filehandle.write('%s\n' % listitem)

# train=train.iloc[ indexes, : ]
# print(len(train))
# train.sort_values(["visitorid","timestamp"], inplace=True)
# print(train[0:20])
times=[]
#dates=[]
for i in train['timestamp']:
    obj_datetime=datetime.datetime.fromtimestamp(i//1000.0)
    #obj_date=obj_datetime.date()
    times.append(obj_datetime)
    #dates.append(obj_date)
train['timestamp_str']=times
#train['date']=dates
#train.sort_values(["visitorid","itemid","date"], inplace=True)
#train.drop_duplicates(subset=['visitorid', 'itemid', "date"], keep="last" ,inplace=True )
print(train[0:20])

train=train.drop('timestamp_str', axis=1)
train=train.drop('match', axis=1)
timeMin = train.timestamp.min()
print("minimum datatime in dataset:")
print(timeMin)
print(datetime.datetime.fromtimestamp(timeMin//1000.0))

print("maximum datatime in dataset:")
timeMax = train.timestamp.max()
print(timeMax)
print(datetime.datetime.fromtimestamp(timeMax//1000.0))

#dt_obj = datetime.strptime('20.12.2016 09:38:42,76','%d.%m.%Y %H:%M:%S,%f')
#split_point = dt_obj.timestamp() * 1000

split_point=datetime.datetime.strptime("2015-09-01 00:00:00.000", '%Y-%m-%d %H:%M:%S.%f').timestamp()
#print(split_point*1000)
#print(datetime.datetime.fromtimestamp(split_point))
#print(datetime.datetime.fromtimestamp(split_point//1000.0))

split_mask= train['timestamp']< split_point*1000
#print(split_mask[0:20])

trainTR = train[split_mask]
#print(len(trainTR))

trainVD = train[~split_mask]
#print(len(trainVD))

#Delete items in training split which are less than 5
trainTR=removeRareItems(trainTR)
#Delete Sessions in training split which are less than 5
trainTR=removeShortSessions(trainTR)

#Delete records in valid split where items are not in training split
trainVD = trainVD[np.in1d(trainVD.itemid, trainTR.itemid)]
#Delete records in valid split where visitors are not in training split
trainVD = trainVD[np.in1d(trainVD.visitorid, trainTR.visitorid)]
#Delete Sessions in valid split which are less than 5
trainVD=removeShortSessions(trainVD)

print('Training Set has', len(trainTR), 'Events, ', trainTR.visitorid.nunique(), 'Sessions, and', trainTR.itemid.nunique(), 'Items\n\n')
trainTR.to_csv(dataAfter + 'retailTrainOnly.txt', sep=',', index=False)
print('Validation Set has', len(trainVD), 'Events, ', trainVD.visitorid.nunique(), 'Sessions, and', trainVD.itemid.nunique(), 'Items\n\n')
trainVD.to_csv(dataAfter + 'retailValid.txt', sep=',', index=False)