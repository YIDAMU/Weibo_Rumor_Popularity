# Weibo_Rumor_Popularity
Dataset for the paper "Predicting and Analyzing the Popularity of False Rumors in Weibo" 

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
https://github.com/YIDAMU/Weibo_Rumor_Popularity.git
cd [this folder]
python3 xxx.py
```
To extract KG representations:
We use open source code via: [https://github.com/autoliuweijie/K-BERT]
