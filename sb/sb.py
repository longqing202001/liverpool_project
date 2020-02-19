import argparse
import numpy as np
import pandas as pd
import keras
import pickle
import keras.backend as K
import ProjectUtility
from time import time
from tqdm import tqdm
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
#from keras.layers import Input, Dense, Dropout, CuDNNGRU, Embedding
from keras.layers import Input, Dense, Dropout, GRU, Embedding

def evaluate_model(model, item_valid_input, valid_labels,num_items, topK):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """  
    hits, ndcgs = [],[]
    target_oh = to_categorical(valid_labels, num_classes=num_items)
    input_oh  = to_categorical(item_valid_input,  num_classes=num_items) 
    input_oh = np.expand_dims(input_oh, axis=1)

    predictions = model.predict(input_oh, batch_size=args.batch_size)
    #predictions = model.predict([np.array(user_valid_input), np.array(item_valid_input)], batch_size=args.batch_size, verbose=0)
    topk_ind = predictions.argsort()[:,::-1][:,:topK]
    for item in zip(topk_ind,valid_labels):
        hr = ProjectUtility.getHitRatio(item[0], item[1])
        ndcg = ProjectUtility.getNDCG(item[0], item[1])
        hits.append(hr)
        ndcgs.append(ndcg)
    hits.append(hr)
    ndcgs.append(ndcg)   
    return (hits, ndcgs)

def get_model(num_items):
    hidden_units = 100
    inputs = Input(batch_shape=(args.batch_size, 1, num_items))
    gru, gru_states = GRU(hidden_units, stateful=True, return_state=True)(inputs)
    drop1 = Dropout(0.25)(gru)
    predictions = Dense(num_items, activation='softmax')(drop1)
    model = Model(input=inputs, output=[predictions])
    return model

def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='session based filtering recommendations')
    parser.add_argument('--base_path', type=str, default='/home/ryu/Desktop/project/liverpoor/')
    #parser.add_argument('--base_path', type=str, default='/content/drive/My Drive/liverpool/')
    parser.add_argument('--train_input_path', type=str, default='data/preprocessed_data/train_input.dat')
    parser.add_argument('--train_mask_path', type=str, default='data/preprocessed_data/train_mask.dat')    
    parser.add_argument('--train_target_path', type=str, default='data/preprocessed_data/train_target.dat')     
    parser.add_argument('--valid_input_path', type=str, default='data/preprocessed_data/valid_input.dat')
    parser.add_argument('--valid_mask_path', type=str, default='data/preprocessed_data/valid_mask.dat')     
    parser.add_argument('--valid_target_path', type=str, default='data/preprocessed_data/valid_target.dat')    
    parser.add_argument('--item_dic_path', type=str, default='data/preprocessed_data/item_dic.txt')
    parser.add_argument('--visitor_dic_path', type=str, default='data/preprocessed_data/visitor_dic.txt')
    parser.add_argument('--item_features_path', type=str, default='data/preprocessed_data/item_features.csv')    
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/')      
    parser.add_argument('--batch_size', type=str, default=256)
    parser.add_argument('--slice_size', type=str, default=256)
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

    #read training input mask data
    with open(args.base_path+args.train_mask_path, 'rb') as f:
        train_mask_inputlist = pickle.load(f)    

    #read training target data
    with open(args.base_path+args.train_target_path, 'rb') as f:
        train_targetlist = pickle.load(f)
    train_targetlist = np.array(train_targetlist)

    #read valid input mask data
    with open(args.base_path+args.valid_mask_path, 'rb') as f:
        valid_mask_inputlist = pickle.load(f)  

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
    model_out_file = args.base_path+args.checkpoint_path+'SB_%d_%d.h5' %(args.latent_dim, time())
    model = get_model(num_items)
    print(model.summary())
    optimizer_obj=keras.optimizers.Adagrad(lr=args.learning_rate)
    model.compile(optimizer=optimizer_obj, loss='categorical_crossentropy',metrics=['accuracy'])
    hr_list = []
    loss_list = []
    early_stop =True
    #training
    for epoch in range(args.epochs):
        with tqdm(total=len(item_train_input)) as pbar:
            print("Training epoch %d. " %(epoch))
            start_index=0
            end_index=start_index+args.slice_size
            count=len(item_train_input)//args.slice_size
            i=0
            while (i<count):
                slice_visitor_train_input=user_train_input[start_index:end_index]
                slice_item_train_input=item_train_input[start_index:end_index]
                slice_item_train_target = train_targetlist[start_index:end_index]
                mask=train_mask_inputlist[i]
                t1 = time()
                # Training
                real_mask = np.ones((args.batch_size, 1))
                for elt in mask:
                    real_mask[elt, :] = 0
                hidden_states = get_states(model)[0]
                hidden_states = np.multiply(real_mask, hidden_states)
                hidden_states = np.array(hidden_states, dtype=np.float32)
                model.layers[1].reset_states(hidden_states)

                train_input = to_categorical(slice_item_train_input, num_classes=num_items) 
                train_input = np.expand_dims(train_input, axis=1)

                train_target = to_categorical(slice_item_train_target, num_classes=num_items)
                tr_loss = model.train_on_batch(train_input, train_target)

                t2 = time()
                start_index+=args.slice_size
                end_index+=args.slice_size
                if end_index>len(user_train_input):
                    end_index=len(user_train_input)
                i+=1
                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss[0]))
                pbar.update(int(args.slice_size))       
        # Evaluation
        (hits, ndcgs) = evaluate_model(model,item_valid_input,valid_targetlist, num_items, args.topK)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean(),
        print("Training epoch %d, Training loss = %.4f,Validation HR@%d = %.4f, NDCG@%d = %.4f. " %(epoch, tr_loss[0], args.topK, hr, args.topK, ndcg))           
        # Using patience to set the early stopping.
        # Always to save the model with minimun loss.
        hr_list.append(hr)
        if hr < np.max(hr_list):
            patience_count += 1
        else:
            patience_count = 0
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            model.save_weights(model_out_file, overwrite=True)
        if (early_stop) and (patience_count == args.patience):
            break
    print("End. Best Iteration %d:  HR@%d  = %.4f, NDCG@%d = %.4f. " %(best_iter, args.topK, best_hr, args.topK, best_ndcg))