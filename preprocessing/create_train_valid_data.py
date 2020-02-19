import lib
import argparse
import torch
import lib
import numpy as np
import os
import datetime
from tqdm import tqdm
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='/home/ryu/Desktop/project/liverpoor/data/preprocessed_data/', type=str)
parser.add_argument('--train_data', default='retailTrainOnly.txt', type=str)
parser.add_argument('--valid_data', default='retailValid.txt', type=str)
parser.add_argument('--train_input_data', default='train_input.dat', type=str)
parser.add_argument('--valid_input_data', default='valid_input.dat', type=str)
parser.add_argument('--train_target_data', default='train_target.dat', type=str)
parser.add_argument('--valid_target_data', default='valid_target.dat', type=str)
parser.add_argument('--train_mask_data', default='train_mask.dat', type=str)
parser.add_argument('--valid_mask_data', default='valid_mask.dat', type=str)
parser.add_argument('--item_dic_data', default='item_dic.txt', type=str)
parser.add_argument('--visitor_dic_data', default='visitor_dic.txt', type=str)
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") 
parser.add_argument('--batch_size', default=256, type=int) 

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def save_pickle_data(dataloader,input_filepath,target_filepath,mask_filepath,batch_size,all_df_data):
    visitorid_s = []
    input_item_s = []
    target_item_s = []
    mask_s = []
    for ii, (visitorid, input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            #for input, target, mask in dataloader:
                visitorid = visitorid.to(device)
                visitorid_s=visitorid_s+visitorid.tolist()
                input = input.to(device)
                input_item_s=input_item_s+input.tolist()              
                target = target.to(device)
                target_item_s=target_item_s+target.tolist()
                mask = mask.to(device)
                mask_s.append(mask.tolist())

    input_items=list(zip(visitorid_s, input_item_s))
    print("input lenghts:"+str(len(input_items)))
    with open(input_filepath, "wb") as f:
        pickle.dump(input_items, f)

    print("mssk lenghts:"+str(len(mask_s)))
    with open(mask_filepath, "wb") as f:
        pickle.dump(mask_s, f)

    print("target lenghts:"+str(len(target_item_s)))
    with open(target_filepath, "wb") as f:
        pickle.dump(target_item_s, f)
     
train_filepath=os.path.join(args.data_folder, args.train_data)
train_df = pd.read_csv(train_filepath, sep=",", usecols=["visitorid", "itemid", "timestamp"])
print(train_df.head())

valid_filepath=os.path.join(args.data_folder, args.valid_data)
#valid_df = pd.read_csv(valid_filepath, sep=",", dtype={'visitorid': int, 'itemid': int, 'timestamp': float})
valid_df = pd.read_csv(valid_filepath, sep=",", usecols=["visitorid", "itemid", "timestamp"])

all_df_data=pd.concat([train_df, valid_df])

item_ids = all_df_data['itemid'].unique()  # type is numpy.ndarray
item2idx = pd.Series(data=np.arange(len(item_ids)),index=item_ids)
itemmap = pd.DataFrame({'itemid': item_ids,'item_idx': item2idx[item_ids].values})

visitor_ids = all_df_data['visitorid'].unique()  # type is numpy.ndarray
visitor2idx = pd.Series(data=np.arange(len(visitor_ids)),index=visitor_ids)
visitormap = pd.DataFrame({'visitorid': visitor_ids,'visitor_idx': visitor2idx[visitor_ids].values})

item_dic_file_path=os.path.join(args.data_folder, args.item_dic_data)
np.savetxt(item_dic_file_path, itemmap, fmt='%d')

visitor_dic_file_path=os.path.join(args.data_folder, args.visitor_dic_data)
np.savetxt(visitor_dic_file_path, visitormap, fmt='%d')

train_data = lib.Dataset(os.path.join(args.data_folder, args.train_data),visitormap=visitormap, itemmap=itemmap)
train_dataloader = lib.DataLoader(train_data, args.batch_size)
train_dataloader.dataset.df.to_csv("./train.csv")

valid_data = lib.Dataset(os.path.join(args.data_folder, args.valid_data),visitormap=visitormap, itemmap=itemmap)
valid_dataloader = lib.DataLoader(valid_data, args.batch_size)
valid_dataloader.dataset.df.to_csv("./valid.csv")

all_df_data=pd.concat([train_dataloader.dataset.df, valid_dataloader.dataset.df])
all_df_data.to_csv("./all.csv")

# create train data
input_filepath=os.path.join(args.data_folder, args.train_input_data)
target_filepath=os.path.join(args.data_folder, args.train_target_data)
mask_filepath=os.path.join(args.data_folder, args.train_mask_data)
save_pickle_data(train_dataloader,input_filepath,target_filepath,mask_filepath,args.batch_size,all_df_data)

#create valid data
input_filepath=os.path.join(args.data_folder, args.valid_input_data)
target_filepath=os.path.join(args.data_folder, args.valid_target_data)
mask_filepath=os.path.join(args.data_folder, args.valid_mask_data)
save_pickle_data(valid_dataloader,input_filepath,target_filepath,mask_filepath,args.batch_size,all_df_data)

