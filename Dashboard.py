#streamlit 
import streamlit as st
import SessionState
import plotly.express as px

# linear algebra
import numpy as np 
import math 

# data processing
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.style.use('ggplot')
sns.set(color_codes=True)

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
import datetime


# Importing sklearn methods
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# import labelencoder
from sklearn.preprocessing import LabelEncoder

# the spearman's correlation between two variables
from scipy.stats import spearmanr

#read data
df=pd.read_csv('C:/Users/SarahAbouIbrahim/Desktop/marketing_data.csv')
print(df.shape)

#cleaning data
def clean_data(df):
    df.rename({' Income ':'Income'}, axis=1, inplace=True)
    df['Income'] = df['Income'].str.replace('$','').str.replace(',','').astype(float)
    return df

print(df.head())

def null_heatmap(df):
    heatmap=sns.heatmap(df.isnull(),yticklabels=False,cmap='YlOrRd');
    return heatmap.figure
def delete_nulls(df):
    df = df[df['Income'].notna()]
    df.columns[df.isnull().any()].tolist()
    return df

def num_plots(df):
    
    list(set(df.dtypes.tolist()))
    df_num = df.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include = ['float64', 'int64'])
    num_plot=df_num.plot(subplots=True, layout=(4,4), kind='box', figsize=(16,18), patch_artist=True,color="navy")
    #plt.subplots_adjust(wspace=0.5)
    return num_plot

def numeric_dist(df):
    list(set(df.dtypes.tolist()))
    df_num = df.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include = ['float64', 'int64'])
    fig, hist = plt.subplots()
    hist=df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8,color='purple');
    #hist.show()
    return hist
def handle_anomalies(df):
    df = df[df['Year_Birth'] > 1910].reset_index(drop=True)
    df = df[df['Income'] < 600000].reset_index(drop=True)
    df = df[~df['Marital_Status'].isin(['Absurd', 'Alone', 'YOLO'])]
    return df
