import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import warnings
from wordcloud import WordCloud
import nltk
from nltk.stem.porter import PorterStemmer 
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/train.txt' ,header=None, sep=";", names = ['text' , 'emotion'])
df = df.reset_index(drop=True)

df.to_csv('./data/train.csv', index=False)
print("Successfully Save in CSV")