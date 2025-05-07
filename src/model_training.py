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
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import json
warnings.filterwarnings('ignore')


df = pd.read_csv('./data/preprocess/train.csv')

stopwords = set(nltk.corpus.stopwords.words('english'))
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

print('Successfully Clean text')


df['text'] = df['text'].apply(clean_text)
lb = LabelEncoder()
df['emotion'] = lb.fit_transform(df['emotion'])


vector = TfidfVectorizer()
X = vector.fit_transform(df['text'])
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print('Successfully done vecorization')


all_metrics = []

Rmb = RandomForestClassifier()
Rmb.fit(X_train,y_train)
pred = Rmb.predict(X_test)

all_metrics.append({
    'Model':'RandomForestClassifier',
    'Accuracy':accuracy_score(y_test,pred)
})




classfier = {
    'Random Classfier':RandomForestClassifier(),
    'MultinomialNB':MultinomialNB(),
    'SVC':SVC()
}

for name , clf in classfier.items():
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    all_metrics.append({
        'Model':f'{name}',
        'Accuracy':accuracy_score(y_test,pred)
    })


lg = LogisticRegression()
lg.fit(X_train,y_train)
pred = lg.predict(X_test)

all_metrics.append({
    'Model':'LogisticRegression',
    'Accuracy':accuracy_score(y_test,pred)
})



with open('./results/training/metrics.json', 'w') as file:
    json.dump(all_metrics, file, indent=4)



import pickle
pickle.dump(lg,open("./src/pickle/logistic_regresion.pkl",'wb'))
pickle.dump(lb,open("./src/pickle/label_encoder.pkl",'wb'))
pickle.dump(vector,open("./src/pickle/vector.pkl",'wb'))

print('Successfully Done ')
