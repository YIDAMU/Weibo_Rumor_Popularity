import re
import pandas as pd
import numpy as np
import sklearn
import transformers
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats.stats import pearsonr
import math
import torch
from tqdm import tqdm
from dataload import Dataset
import random
from clfkg_test import CLFKG
from clfatt_test import CLFATTKG
from clfmax_test import CLFKGMax
from clfmean_test import CLFKGMEAN
from torch.optim import Adam
from transformers.optimization import AdamW
import torch.nn as nn
#from train_eval import train, evaluate
#
d1 = pd.read_csv('data_final.csv', lineterminator='\n')
#
tr, df_test = train_test_split(d1, test_size=0.1, random_state=777)
###
df_train, df_val = train_test_split(tr, test_size=0.11111, random_state=777)
###
import torch
from torch import nn
from transformers import BertModel
import math
from torch.optim import Adam
from transformers.optimization import AdamW
from tqdm import tqdm
from dataload import Dataset
def train(model, train_data, val_data, learning_rate, epochs):
    global best_loss
    train_data, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)    
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32)
    use_cuda = torch.cuda.is_available()    
    device = torch.device("cuda" if use_cuda else "cpu")
#     if torch.cuda.is_available():
#         device = "cuda:1"
    #### MSE loss for regression
    criterion = nn.MSELoss()    
    #optimizer = Adam(model.parameters(), lr= learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = AdamW(model.parameters(), lr= learning_rate)        
    for epoch_num in range(epochs):
            total_loss_train = 0
            for train_input, train_descripton_input, train_profile, train_kg, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)                
                mask_text = train_input['attention_mask'].squeeze(1).to(device)
                mask_desc = train_descripton_input['attention_mask'].squeeze(1).to(device)
                input_id_text = train_input['input_ids'].squeeze(1).to(device) 
                input_id_desc = train_descripton_input['input_ids'].squeeze(1).to(device)
                user_features = train_profile.to(device)
                train_kgs = train_kg.to(device)
                output = model(
                    input_id_text = input_id_text, 
                    input_id_desc = input_id_desc, 
                    mask_text = mask_text, 
                    mask_desc = mask_desc, 
                    user_features = user_features,
                    kg_features=train_kgs
                ) 
                batch_loss = criterion(output, train_label.unsqueeze(1).float())                
                total_loss_train += batch_loss.item()           
                model.zero_grad()
                batch_loss.backward()                
                optimizer.step()
            ########## eval on dev set
            total_loss_val = 0
            with torch.no_grad():
                for dev_input, dev_description_input, dev_profile, dev_kg, dev_label in val_dataloader:
                    dev_label = dev_label.to(device)                
                    mask_text = dev_input['attention_mask'].squeeze(1).to(device)
                    mask_desc = dev_description_input['attention_mask'].squeeze(1).to(device)
                    input_id_text = dev_input['input_ids'].squeeze(1).to(device) 
                    input_id_desc = dev_description_input['input_ids'].squeeze(1).to(device)
                    user_features = dev_profile.to(device)
                    dev_kgs = dev_kg.to(device)
                    output = model(
                        input_id_text = input_id_text, 
                        input_id_desc = input_id_desc, 
                        mask_text = mask_text, 
                        mask_desc = mask_desc, 
                        user_features = user_features,
                        kg_features = dev_kgs
                    )
                    batch_loss = criterion(output, dev_label.unsqueeze(1).float())
                    total_loss_val += batch_loss.item()
                ####    
                val_loss=total_loss_val/(len(df_val)/32)
                #print(val_loss)
                if val_loss < best_loss:
                    best_loss=val_loss
                    #print(best_loss)
                    torch.save(model.state_dict(), 'best_model_checkpoint')
                    #print('saved')
                    
# EPOCHS = 4
# model = CLFATT()
# LR = 2e-5              
# train(model, df_train, df_val, LR, EPOCHS)
### predictionssss
def evaluate(model, test_data):
    mse=[]
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32)
    use_cuda = torch.cuda.is_available()    
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    #total_acc_test = 0    
    with torch.no_grad():
        for test_input, test_description_input, test_profile, test_kg, test_label in test_dataloader:
            test_label = test_label.to(device)                
            mask_text = test_input['attention_mask'].squeeze(1).to(device)
            mask_desc = test_description_input['attention_mask'].squeeze(1).to(device)
            input_id_text = test_input['input_ids'].squeeze(1).to(device) 
            input_id_desc = test_description_input['input_ids'].squeeze(1).to(device)
            user_features = test_profile.to(device)
            test_kg   = test_kg.to(device)
            output = model(
                input_id_text = input_id_text, 
                input_id_desc = input_id_desc, 
                mask_text = mask_text, 
                mask_desc = mask_desc, 
                user_features = user_features,
                kg_features=test_kg
            )
            #mse = math.sqrt(mean_squared_error(test_label, output.cpu()))
            #        total_acc_test += mse
            mse.append(output.cpu())
    return mse
########### Attention


def pear(mse):
    new=[]
    for i in mse:
        for ii in i:
            new.append(ii.tolist()[0])
    return new
############

seeds=[111,222,333]
p1=[]
p2=[]
p3=[]
for sd in seeds:
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    best_loss = 1000
    EPOCHS = 2
    #model = ['CLFKG', 'CLFATTKG', 'CLFKGMax', 'CLFKGMEAN']
    model = CLFKG()
    LR = 2e-5              
    train(model, df_train, df_val, LR, EPOCHS)
    bestmodel = CLFKG()
    # load best checkpoint
    bestmodel.load_state_dict(torch.load('best_model_checkpoint'))
    mse=evaluate(bestmodel, df_test)
    new=[]
    new=pear(mse)      
    print(math.sqrt(mean_squared_error(df_test.labels, new)))
    print(pearsonr(df_test.labels.values, new))
    p1.append(math.sqrt(mean_squared_error(df_test.labels.values, new)))
    p2.append(pearsonr(df_test.labels.values, new)[0])
    print(sd)
    cc=mean_absolute_error(df_test.labels.values, new)
    p3.append(cc)
    print(cc)
print((np.round(np.mean(p1),3), np.round(np.std(p1),3)))
print((np.round(np.mean(p2),3), np.round(np.std(p2),3)))
print((np.round(np.mean(p3),3), np.round(np.std(p3),3)))       
    
    
