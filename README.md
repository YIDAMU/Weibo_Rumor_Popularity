# Weibo_Rumor_Popularity
Dataset for the paper "Predicting and Analyzing the Popularity of False Rumors in Weibo" 

We conduct our experiments on a single Nvidia-A100 (40GB)
```text
python == 3.10.9
torch == 2.0.1+cu117
transformers == 4.33.3
```

To explore datasets:
```python
#raw dataset
import pandas as pd
data=pd.read_csv('Weibo_Rumor_Popularity_raw.tsv', sep='\t', lineterminator='\n')
#pre-processed dataset
data=pd.read_csv('Weibo_Rumor_Popularity_clean.csv', lineterminator='\n')
```
To run experiments:
```bash
git clone https://github.com/YIDAMU/Weibo_Rumor_Popularity.git
#extract data
cd [this_folder]
CUDA_VISIBLE_DEVICES= "Your_GPU" python3 run_model.py
```
To extract KG representations:
We employ a knowledge_BERT pipeline via: [https://github.com/autoliuweijie/K-BERT]
