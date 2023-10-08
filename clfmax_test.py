import torch
from torch import nn
from transformers import BertModel
import math
class DotAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DotAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_dim, 1, bias = False)
        stdv = 1. / math.sqrt(hidden_dim)
        self.attn1.weight.data.uniform_(-stdv,stdv)
    def forward(self, hidden):
        attn1 = (self.attn1(hidden) / (hidden.shape[-1]) ** 0.5).squeeze(-1)
        #attn1.masked_fill_(~mask.bool(), -float('inf'))
        weights = torch.softmax(attn1, dim = -1)
        return weights

class CLFKGMax(nn.Module):    
    def __init__(self, dropout=0.1):
        super(CLFKGMax, self).__init__()
        #### hfl/chinese-bert-wwm 
        #### for Rumour TexT
        self.bert_text = BertModel.from_pretrained('./outputs/plm_task_specific_144_Weibo')   
        self.bert_kg = nn.Linear(768, 128)     
        #### for Description
        self.bert_desc = BertModel.from_pretrained('./outputs/plm_task_specific_144_Weibo')        
        ### for USER FEATURES
        self.linear_user = nn.Linear(12, 128)
        self.linear_reduce_1 = nn.Linear(768, 128) 
        self.linear_reduce_2 = nn.Linear(768, 128)      
        ### for Mapping all Fetures together
        self.transform = nn.Linear(128*4, 128)
        self.relu = nn.ReLU() 
        self.attention = DotAttention(128)              
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, 1)
    def forward(self, input_id_text, input_id_desc, mask_text, mask_desc, user_features, kg_features):
        _, pooled_text = self.bert_text(input_ids= input_id_text, attention_mask=mask_text, return_dict=False)
        _, pooled_desc = self.bert_desc(input_ids= input_id_desc, attention_mask=mask_desc, return_dict=False)        
        #_, pooled_kg = self.bert_kg(input_ids= input_id_text, attention_mask=mask_text, return_dict=False)
        user_feature = self.linear_user(user_features)  
        kgs =  self.bert_kg(kg_features)     
        pooled_text = self.dropout(pooled_text)
        pooled_text = self.linear_reduce_1(pooled_text)
        pooled_desc = self.dropout(pooled_desc)
        pooled_desc = self.linear_reduce_2(pooled_desc)
        combined = torch.stack([pooled_text, pooled_desc, user_feature, kgs],1)
        combined = combined.max(1)[0] 
        #pooled_kg = self.dropout(pooled_kg)
        #pooled_kg = self.linear_reduce_2(pooled_kg)	
        #combined = self.transform(torch.cat([pooled_text, pooled_desc, user_feature, kgs], dim = -1))
	## attention over combined features 
        #torch.Size([BS, 3, 128])       
        #attention_weights = self.attention(comb_features)
        #transformed = self.relu(self.user_drop(transformed))            
        ## combine all features
        #combined = self.transform(comb_features)  # Batch_size x (768*2 + 128) --> B X 768   
        #combined = (attention_weights.unsqueeze(-1) * comb_features).sum(1)
        #relus= self.relu(combined)    
        ## now for the classifier    
        #dropout_output = self.dropout(combined)
        linear_output = self.classifier(combined)
        #final_layer = self.relu(linear_output)
        return linear_output
