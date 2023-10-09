# Weibo_Rumor_Popularity
Dataset for the paper "Predicting and Analyzing the Popularity of False Rumors in Weibo" 

We conduct experiments on a single Nvidia-A100 (40GB). Details:
```text
python == 3.10.9
torch == 2.0.1+cu117
transformers == 4.33.3
```

To explore datasets:
```python
#raw dataset from Weibo API
import pandas as pd
data=pd.read_csv('Weibo_Rumor_Popularity_Raw.tsv', sep='\t', lineterminator='\n')
#pre-processed dataset for experiments
#extract from data_final.zip
data=pd.read_csv('data_final.csv', lineterminator='\n')
```

To use our BERT_Weibo_Rumor model via Huggingface:
```python
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("YidaM4396/BERT_Weibo_Rumor")
model = BertForSequenceClassification.from_pretrained("YidaM4396/BERT_Weibo_Rumor")
```

To run experiments:
```bash
git clone https://github.com/YIDAMU/Weibo_Rumor_Popularity.git
#extract data from data_final.zip
cd [this_folder]
CUDA_VISIBLE_DEVICES= "Your_GPU" python3 run_model.py
```
To extract KG representations:
We employ a knowledge_BERT pipeline via: [https://github.com/autoliuweijie/K-BERT]
