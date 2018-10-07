from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
import sqlite3
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

# conn = sqlite3.connect('pitchfork.sqlite')
# c = conn.cursor()

stop_words = set(stopwords.words('english') + list(string.punctuation))

def blobbing():
    # df = pd.read_sql_query("select c.reviewid, c.content, r.score from content c, reviews r where c.reviewid = r.reviewid", conn)
    #
    # sentiment = []
    #rounded_scores = []

    with open('dataframe', 'rb') as file:
        df = pickle.load(file)

    # for index, row in df.iterrows():
    #     rounded_scores.append(round(row.score))
    #
    # df['rounded_score'] = rounded_scores
    #
    #
    # df['sentiment'] = sentiment
    # df['processed'] = df.content.apply(process, stop_words)
    # df = df[df['processed'].notnull()]
    #
    # with open('dataframe', 'wb') as file:
    #     pickle.dump(df, file)

    #print(df.head(10))

    train, test = train_test_split(df, test_size=0.2)


    # json_dump = []
    # train = 1
    # for index, row in df.iterrows():
    #     if train <= 10:
    #         json_dump.append({'text': row['content'], 'label': row['sentiment']})
    #     train += 1
    #
    #
    # json_string = json.dumps(json_dump)
    # with open('sentiment.json', 'w') as file:
    #     file.write(json_string)

    # small_train, small_test = [], []
    #
    # for i in range(500):
    #     small_train.append((train.iloc[i]['content'], train.iloc[i]['sentiment']))
    # #
    # for i in range(500):
    #     small_test.append((test.iloc[i]['content'], test.iloc[i]['sentiment']))

    # with open('sentiment.json', 'r') as file:
    #     cl = NaiveBayesClassifier(file, format='json')

    # cl = NaiveBayesClassifier(small_train)
    # with open('NaiveBayes', 'wb') as file:
    #     pickle.dump(cl, file)

    # with open('NaiveBayes', 'rb') as file:
    #     cl = pickle.load(file)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train['processed'].tolist())
    X_test = vectorizer.transform(test['processed'].tolist())
    y_train = train['rounded_score'].tolist()
    y_test = test['rounded_score'].tolist()

    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    lr.fit(X_train, y_train)
    predictor = lr.predict(X_test)
    # for i in range(20):
    #     print(test.iloc[i]['reviewid'], test.iloc[i]['score'], predictor[i])
    print('Logistic regression =', lr.score(X_test, y_test))


    # nb = MultinomialNB()
    # nb.fit(X_train, y_train)
    # print('Naive Bayes =', nb.score(X_test, y_test))



    conf_mat = confusion_matrix(y_test, predictor)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=df.rounded_score.values, yticklabels=df.rounded_score.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # prob_dist = cl.prob_classify((df.iloc[14000]['content']))
    # print(prob_dist.max(), prob_dist.prob('pos'), prob_dist.prob('neg'))
    # print(cl.accuracy(small_test))

    # blob = TextBlob(df.iloc[4]['content'])
    # print(blob.sentiment)
def process(text):
    tokens = TextBlob(text).words
    tokens = [t.lemmatize() for t in tokens if t not in stop_words]
    # tokens = [re.sub(re_punct, '', t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2]
    if len(tokens) > 0:
        return ' '.join(tokens)
    else:
        return None

blobbing()