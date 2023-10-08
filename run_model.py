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
from train_eval import train, evaluate
#
d1 = pd.read_csv('data_final.csv', lineterminator='\n')
#
tr, df_test = train_test_split(d1, test_size=0.1, random_state=777)
###
df_train, df_val = train_test_split(tr, test_size=0.11111, random_state=777)
###


def pear(mse):
    new=[]
    for i in mse:
        for ii in i:
            new.append(ii.tolist()[0])
    return new
############

seeds=[555,777,999]
p1=[]
p2=[]
p3=[]
for sd in seeds:
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    best_loss = 1000
    EPOCHS = 4
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
    #math.sqrt(mean_squared_error(df_test.labels, new))
    #print(pearsonr(df_test.labels.values, new))
    p1.append(math.sqrt(mean_squared_error(df_test.labels.values, new)))
    p2.append(pearsonr(df_test.labels.values, new)[0])
    #print(sd)
    cc=mean_absolute_error(df_test.labels.values, new)
    p3.append(cc)
    #print(cc)
print((np.round(np.mean(p1),3), np.round(np.std(p1),3)))
print((np.round(np.mean(p2),3), np.round(np.std(p2),3)))
print((np.round(np.mean(p3),3), np.round(np.std(p3),3)))       
    
    
