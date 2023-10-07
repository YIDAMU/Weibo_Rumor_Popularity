# Weibo_Rumor_Popularity
Dataset for the paper "Predicting and Analyzing the Popularity of False Rumors in Weibo" 


'''python
#raw dataset
import pandas as pd
data=pd.read_csv('Weibo_Rumor_Popularity_raw.tsv', sep='\t', lineterminator='\n')
#pre-processed dataset
data=pd.read_csv('Weibo_Rumor_Popularity_clean.csv', lineterminator='\n')

To run experiments:
'''python
cd [this folder]
python3 xxx.py
