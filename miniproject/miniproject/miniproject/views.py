import string
from tokenize import String
from django.shortcuts import render
import numpy as np
import pandas as pd
import gradio as gr
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pyparsing import Char
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk



def home(request):
    return render(request,"home.html")

def predict(request):
    return render(request,"predict.html")
    
def result(request):
    # Loading the Dataset into pandas Dataframe
    news_dataset = pd.read_csv(r'C:\Users\Dell\Desktop\miniproject\train.csv')
    # Replacaing the null values with empty strings
    news_dataset = news_dataset.fillna('')
    # Merging the author name and news title
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    # Seperating the data and Label
    X = news_dataset.drop(columns='label', axis=1)
    Y = news_dataset['label']
    port_stem = PorterStemmer()
    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]',' ', content)
        stemmed_content = stemmed_content.lower()   
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] 
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    news_dataset['content'] =  news_dataset['content'].apply(stemming)
    X = news_dataset['content'].values
    Y = news_dataset['label'].values
 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=2)
    # transform the text data to feature vectors that can be used as input to the Logistic regression

    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # convert Y_train and Y_test values as integers

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    X_train_prediction = model.predict(X_train_features)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = model.predict(X_test_features)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    var1 =(request.GET['n1'])
    var2 =(request.GET['n2'])
    def news_check(var1, var2):
        
        res=""
    # convert text to feature vectors
        input_mail = [var1 + ' ' + var2]
        input_data_features = feature_extraction.transform(input_mail)

    # making prediction

        prediction = model.predict(input_data_features)

        if(prediction[0] == 0):
            return "The news is Real!"
        else:
            return "The news is Fake..."

    return render(request,"predict.html",{"result2":news_check(var1,var2)})