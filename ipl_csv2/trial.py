
import pandas as pd
import os
import ast
import json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

    # Importing the initial all matches dataset
dataset = pd.read_csv(r'D:\cricket score analysis\ipl_csv2\all_matches.csv')  # ye main file 
# transforming dataset such that we have 2 innings and overall 6 over , also  total runs in 6 over

dataset = dataset[dataset['ball'] < 6.0]    
dataset = dataset[dataset['innings'] < 3]  
dataset['total_runs'] = dataset['runs_off_bat'] + dataset['extras']
print(dataset.head())
dataset = dataset.groupby(['match_id', 'venue', 'innings', 'batting_team','striker','non_striker','bowler', 'bowling_team','total_runs'], as_index=False).agg(total_runs = ('total_runs','sum')).reset_index()

#dataset.to_csv(r'to_be_trained.csv', index=False)
print(dataset.head())

dataset = dataset.groupby(['match_id', 'venue', 'innings', 'batting_team','striker','non_striker','bowler', 'bowling_team','total_runs'], as_index=False).agg(total_runs = ('batting_team','sum')).reset_index()

#dataset.to_csv(r'to_be_trained.csv', index=False)
print(dataset.head())