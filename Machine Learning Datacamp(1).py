#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import time 
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
english_stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import for the Machine Learning

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import classification_report

import seaborn as sns


# In[3]:


train_df=pd.read_csv('twcs.csv')
train_df.head(20)


# In[4]:


train_df["in_response_to_tweet_id"].fillna(0, inplace=True)
train_df["in_response_to_tweet_id"] = train_df["in_response_to_tweet_id"].astype(int)
train_df.head(20)


# In[5]:


duplicated_rows = []

# Iterate over each row in the DataFrame
for index, row in train_df.iterrows():
    # Get the values from the response_tweet_id column and split them by commas
    response_ids = str(row['response_tweet_id']).split(',')
    if response_ids:
        train_df.at[index, 'response_tweet_id'] = response_ids[0] 
    
    # Check if there are multiple values in either column
    if len(response_ids) > 1 :
        # Duplicate the row for each value in response_tweet_id and in_response_to_tweet_id
        for i in range(1,len(response_ids)):
                       #response_id in response_ids:
            # Create a copy of the row and update the respective columns
            duplicated_row = row.copy()
            duplicated_row['response_tweet_id'] = response_ids[i]
                
            # Append the duplicated row to the list
            duplicated_rows.append(duplicated_row)

# Create a new DataFrame from the duplicated rows
df2 = pd.DataFrame(duplicated_rows)
df2.head(20)


# In[6]:


train_df.head(20)


# In[7]:


# Concatenate the original DataFrame with the duplicated rows DataFrame
df_combined = pd.concat([train_df, df2], ignore_index=True)
df_combined.sort_values('tweet_id', ascending=True, inplace=True)
df_combined.reset_index(inplace= True)
# Print the resulting DataFrame
df_combined.head(20)


# In[8]:


del df_combined['index']
print(df_combined.columns)


# In[9]:


df_combined


# In[10]:


# Create a new DataFrame, df2, by filtering rows based on author_id
df_query = df_combined[df_combined['inbound']== True]
#train_df[pd.to_numeric(train_df['author_id'], errors='coerce').notnull()]
df_reponse = df_combined[df_combined['inbound']== False]
# Print the resulting DataFrame
df_query


# In[11]:


df_reponse


# In[12]:


import numpy as np
df_query.replace("nan", 0, inplace=True)
df_query["response_tweet_id"] = df_query["response_tweet_id"].astype(int)
df_query.info()


# In[13]:


merged_df = pd.merge(df_query, df_reponse, left_on='response_tweet_id', right_on='tweet_id', how='left')
merged_df


# In[14]:


del merged_df['in_response_to_tweet_id_x']
del merged_df['in_response_to_tweet_id_y']
merged_df


# In[15]:


df_final = merged_df[['text_x','text_y']]
df_final = df_final.dropna()


# In[16]:


df_final['text_x'] = df_final['text_x'].str.replace(r'@\w+\s*', '', regex=True)

# Remove words starting with "@" from text_y column
df_final['text_y'] = df_final['text_y'].str.replace(r'@\w+\s*', '', regex=True)


# In[17]:


df_final.columns = ['query','response']
df_final


# In[25]:


df_sample = df_final.sample(frac = 0.5)
df_sample.to_csv("sample_tweets-sav.csv")


# In[19]:


nltk.download('punkt')

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation and non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply the preprocessing to the 'query' column of your data frame
df_sample['preprocessed_query'] = df_sample['query'].apply(preprocess_text)


# In[21]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_sample['preprocessed_query'])


# In[22]:


responses = df_sample['response'].tolist()


# In[23]:


similarity_matrix = cosine_similarity(X)

