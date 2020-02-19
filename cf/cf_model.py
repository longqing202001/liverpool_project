import math
import pandas as pd
import numpy as np
import os
import pickle
import random
import re
import keras
import argparse
import ProjectUtility
from tqdm import tqdm
from time import time
from keras.layers import Input
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.core import Reshape, Flatten
from keras.layers.merge import Multiply, multiply, Concatenate
from keras.layers.merge import Dot
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Adadelta
from keras import backend as K
from keras.utils import plot_model
import matplotlib.pyplot as plt

def evaluate_model(model, user_valid_input, item_valid_input, valid_labels, num_items, topK):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each batch.
    """ 
    print('\n# Evaluate on test data')
    slice_y_val = keras.utils.to_categorical(valid_labels, num_items)
    evaluate_loss = model.evaluate([np.array(user_valid_input), np.array(item_valid_input)],np.array(slice_y_val),batch_size=args.batch_size)
    hits, ndcgs = [],[]
    predictions = model.predict([np.array(user_valid_input), np.array(item_valid_input)], batch_size=args.batch_size, verbose=0)
    print(len(predictions))
    topk_ind = predictions.argsort()[:,::-1][:,:topK]
    for item in zip(topk_ind,valid_labels):
        hr = ProjectUtility.getHitRatio(item[0], item[1])
        ndcg = ProjectUtility.getNDCG(item[0], item[1])
        hits.append(hr)
        ndcgs.append(ndcg)
    hits.append(hr)
    ndcgs.append(ndcg)  
    return (hits, ndcgs, evaluate_loss)

def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    ### define placeholder.
    user_id_input = Input(shape=[1], name='user')
    item_id_input = Input(shape=[1], name='item')

    ### define embedding size and layers.

    user_embedding = Embedding(output_dim = latent_dim, input_dim = num_users,
                               input_length=1, name='user_embedding',
                               embeddings_regularizer = l2(regs[0]))(user_id_input)
    item_embedding = Embedding(output_dim = latent_dim, input_dim = num_items,
                               input_length=1, name='item_embedding',
                              embeddings_regularizer = l2(regs[1]))(item_id_input)

    user_vecs = Reshape([latent_dim])(user_embedding)
    item_vecs = Reshape([latent_dim])(item_embedding)

    concat = Concatenate()([user_vecs, item_vecs])
    concat_dropout = keras.layers.Dropout(0.25)(concat)
    out = Dense(num_items, activation='softmax')(concat_dropout)
    model = Model(inputs=[user_id_input, item_id_input], outputs=out)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='collaborative filtering recommendations')
    #parser.add_argument('--base_path', type=str, default='/home/ryu/Desktop/project/liverpoor/')
    parser.add_argument('--base_path', type=str, default='/content/drive/My Drive/liverpool/')    
    parser.add_argument('--train_input_path', type=str, default='data/preprocessed_data/train_input.dat')
    parser.add_argument('--train_mask_path', type=str, default='data/preprocessed_data/train_mask.dat')    
    parser.add_argument('--train_target_path', type=str, default='data/preprocessed_data/train_target.dat')     
    parser.add_argument('--valid_input_path', type=str, default='data/preprocessed_data/valid_input.dat')
    parser.add_argument('--valid_mask_path', type=str, default='data/preprocessed_data/valid_mask.dat')     
    parser.add_argument('--valid_target_path', type=str, default='data/preprocessed_data/valid_target.dat')    
    parser.add_argument('--item_dic_path', type=str, default='data/preprocessed_data/item_dic.txt')
    parser.add_argument('--visitor_dic_path', type=str, default='data/preprocessed_data/visitor_dic.txt')
    parser.add_argument('--item_features_path', type=str, default='data/preprocessed_data/item_features.csv')
    parser.add_argument('--train_loss_path', type=str, default='result/cf_loss.png')  
    parser.add_argument('--evaluation_path', type=str, default='result/cf_evaluation.png')  
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/')
    parser.add_argument('--batch_size', type=str, default=100)
    parser.add_argument('--slice_size', type=str, default=10000)
    parser.add_argument('--topK', type=str, default=50)
    parser.add_argument('--epochs', type=str, default=100)
    parser.add_argument('--latent_dim', type=str, default=128)
    parser.add_argument('--learning_rate', type=str, default=0.1)
    parser.add_argument('--patience', type=str, default=5)
    args = parser.parse_args()

    visitor_dic=ProjectUtility.readItemDic(args.base_path+args.visitor_dic_path)
    num_users=len(visitor_dic)

    item_dic=ProjectUtility.readItemDic(args.base_path+args.item_dic_path)
    num_items=len(item_dic)

    #read training input data
    with open(args.base_path+args.train_input_path, 'rb') as f:
        train_inputlist = pickle.load(f)
    train_inputlist = np.array(train_inputlist)     
    user_train_input=train_inputlist[:,0]
    item_train_input=train_inputlist[:,1]

    #read training target data
    with open(args.base_path+args.train_target_path, 'rb') as f:
        train_targetlist = pickle.load(f)
    train_targetlist = np.array(train_targetlist)

    #read validation input data
    with open(args.base_path+args.valid_input_path, 'rb') as f:
        valid_inputlist = pickle.load(f)
    valid_inputlist = np.array(valid_inputlist)     
    user_valid_input=valid_inputlist[:,0]
    item_valid_input=valid_inputlist[:,1]

    #read validation target data
    with open(args.base_path+args.valid_target_path, 'rb') as f:
        valid_targetlist = pickle.load(f)    
    valid_targetlist = np.array(valid_targetlist)    

    verbose = 0
    best_hr, best_ndcg, best_iter = -1, -1, -1
    model_out_file = args.base_path+args.checkpoint_path+'CF_%d_%d.h5' %(args.latent_dim, time())
    model = get_model(num_users, num_items, args.latent_dim, regs=[0,0])
    #plot_model(model, to_file='model.png')
    print(model.summary())
    optimizer_obj=keras.optimizers.Adagrad(lr=args.learning_rate)
    model.compile(optimizer=optimizer_obj, loss='categorical_crossentropy',metrics=['accuracy'])
    hr_list = []
    ndcg_list =[]
    loss_list = []
    early_stop =True
    train_losses=[]
    evaluation_losses=[]
    #training
    for epoch in range(args.epochs):
        with tqdm(total=len(item_train_input)) as pbar:
            print("Training epoch %d. " %(epoch))
            start_index=0
            slice_size=int(len(user_train_input)/128*4)
            slice_size=args.slice_size            
            end_index=start_index+slice_size
            count=len(user_train_input)//slice_size
            i=0
            while (i<count):
                slice_user_train_input=user_train_input[start_index:end_index]
                slice_item_train_input=item_train_input[start_index:end_index]
                slice_y_train = keras.utils.to_categorical(train_targetlist[start_index:end_index], num_items)  
                t1 = time()
                # Training
                hist = model.fit([np.array(slice_user_train_input), np.array(slice_item_train_input)], #input
                         np.array(slice_y_train), # labels 
                         batch_size = args.batch_size,
                         epochs = 1, verbose = verbose, shuffle = True)
                t2 = time()
                start_index+=slice_size
                end_index+=slice_size
                if end_index>len(user_train_input):
                    end_index=len(user_train_input)
                i+=1
                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, hist.history['loss'][0]))
                pbar.update(int(slice_size))          
        # Evaluation
        train_losses.append(hist.history['loss'][0])
        (hits, ndcgs,evaluation_loss) = evaluate_model(model, user_valid_input, item_valid_input, valid_targetlist, num_items, args.topK)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean(),
        print("Training epoch %d, Training loss = %.4f,Validation HR@%d = %.4f, NDCG@%d = %.4f. " %(epoch, hist.history['loss'][0], args.topK, hr, args.topK, ndcg))            
        # Using patience to set the early stopping.
        # Always to save the model with minimun loss.
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        if hr < np.max(hr_list):
            patience_count += 1
        else:
            patience_count = 0
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            model.save_weights(model_out_file, overwrite=True)
        if (early_stop) and (patience_count == args.patience):
            break
    print("End. Best Iteration %d:  HR@%d  = %.4f, NDCG@%d = %.4f. " %(best_iter, args.topK, best_hr, args.topK, best_ndcg))



