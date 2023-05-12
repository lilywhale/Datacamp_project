#!/usr/bin/env python
# coding: utf-8

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import demoji
import warnings
warnings.filterwarnings('ignore')
import re, nltk

from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from  sklearn import datasets
from langdetect import detect
import emoji

#loading the dataset and doing preprocessing 
df_init = pd.read_csv("python_projects/sample_tweets_sav.csv")

df = df_init.sample(frac = 0.005)

del df['Unnamed: 0']


df['query'] = df.loc[:,'query'].str.lower()
df['response'] = df.loc[:,'response'].str.lower()



print("\n the hard part \n")

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False
    

def remove_urls(text):
    url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
    text = url_pattern.sub('', text)
    return text

# Remove rows that are not in English
df = df[df['query'].apply(is_english)]
print("english done ")
print(df.info())




df['query'] = df['query'].astype(str)
df['response'] = df['response'].astype(str)


print("stop words")
def removestopwords(data):
    removedstopwords = []
    for review in data:
        removedstopwords.append(
            ' '.join([word for word in review.split()  if word not in english_stop_words]))
    return removedstopwords

df['text'] = df['query'] + ' ' + df['response']

text = df['text']

nostopwords_text = removestopwords(text)



# Combine the query and response columns into a single text column
print(" vectorising ")

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(nostopwords_text)


#Détermination de la valeur optimale de K
#On a afficher la courbe avec un autre scrpit afin de voir la progression on a choisi K=47 car la courbe semblait converger apres ce point
"""tab =[]
for i in range(0,55):
    print(i)
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X)
    tab.append(kmeans.inertia_)
    print(tab)
    plt.plot(range(1, i+1), tab)
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()"""

print(" fiting ")

#K-means clustering
k = 47  
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_

#Creation des dataframes correspondant a chaque label
df2 = pd.DataFrame()
df2['query'] = df['query']
df2['response'] = df['response']
df2['labels'] = labels

df_list = []
df_list.append(df2[df2['labels'] == 0])
df_list.append(df2[df2['labels'] == 1])
df_list.append(df2[df2['labels'] == 2])
df_list.append(df2[df2['labels'] == 3])
df_list.append(df2[df2['labels'] == 4])
df_list.append(df2[df2['labels'] == 5])
df_list.append(df2[df2['labels'] == 6])
df_list.append(df2[df2['labels'] == 7])
df_list.append(df2[df2['labels'] == 8])
df_list.append(df2[df2['labels'] == 9])
df_list.append(df2[df2['labels'] == 10])
df_list.append(df2[df2['labels'] == 11])
df_list.append(df2[df2['labels'] == 12])
df_list.append(df2[df2['labels'] == 13])
df_list.append(df2[df2['labels'] == 14])
df_list.append(df2[df2['labels'] == 15])
df_list.append(df2[df2['labels'] == 16])
df_list.append(df2[df2['labels'] == 17])
df_list.append(df2[df2['labels'] == 18])
df_list.append(df2[df2['labels'] == 19])
df_list.append(df2[df2['labels'] == 20])
df_list.append(df2[df2['labels'] == 21])
df_list.append(df2[df2['labels'] == 22])
df_list.append(df2[df2['labels'] == 23])
df_list.append(df2[df2['labels'] == 24])
df_list.append(df2[df2['labels'] == 25])
df_list.append(df2[df2['labels'] == 26])
df_list.append(df2[df2['labels'] == 27])
df_list.append(df2[df2['labels'] == 28])
df_list.append(df2[df2['labels'] == 29])
df_list.append(df2[df2['labels'] == 30])
df_list.append(df2[df2['labels'] == 31])
df_list.append(df2[df2['labels'] == 32])
df_list.append(df2[df2['labels'] == 33])
df_list.append(df2[df2['labels'] == 34])
df_list.append(df2[df2['labels'] == 35])
df_list.append(df2[df2['labels'] == 36])
df_list.append(df2[df2['labels'] == 37])
df_list.append(df2[df2['labels'] == 38])
df_list.append(df2[df2['labels'] == 39])
df_list.append(df2[df2['labels'] == 40])
df_list.append(df2[df2['labels'] == 41])
df_list.append(df2[df2['labels'] == 42])
df_list.append(df2[df2['labels'] == 43])
df_list.append(df2[df2['labels'] == 44])
df_list.append(df2[df2['labels'] == 45])
df_list.append(df2[df2['labels'] == 46])

# On stock chaque groupe de requettes dans un dictionnaire qui sera ensuite ajouter a une liste 

def create_intent_dict(df_list):
    intents = []
    for i, df in enumerate(df_list):
        patterns = df['query'].tolist()
        responses = df['response'].tolist()
        intent = {
            "tag": str(i),
            "patterns": patterns,
            "responses": responses
        }
        intents.append(intent)
    return intents

intents = create_intent_dict(df_list)
result = {"intents": intents}


# Conversion en json et ecriture dans un fichier .json 
json_data = json.dumps(result, indent=4)

with open("C:/Users/mimil/programmation/python_projects/chatbot_project/intents_final1.json", "w") as file:
    file.write(json_data)


