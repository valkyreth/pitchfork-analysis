from textblob import TextBlob
import pandas as pd
import sqlite3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import string
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import io
import base64

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

# conn = sqlite3.connect('pitchfork.sqlite')
# c = conn.cursor()

stop_words = set(stopwords.words('english') + list(string.punctuation))

def blobbing():
    print('start process', time.asctime())
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

    train, test = train_test_split(df, test_size=0.25, stratify=df.rounded_score, random_state=1)


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
    # print('starting vectorisation', time.asctime())
    # #vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=5, max_df=0.5)
    # #X_train = vectorizer.fit_transform(train['processed'].tolist())
    with open('X_train', 'rb') as file, open('vectorizer(1)', 'rb') as vfile:
        X_train = pickle.load(file)
        vectorizer = pickle.load(vfile)
    X_test = vectorizer.transform(test['processed'].tolist())
    #y_train = train['rounded_score'].tolist()
    y_test = test['rounded_score'].tolist()
    print('starting model', time.asctime())

    # lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    # lr.fit(X_train, y_train)
    with open('LogisticRegression (newton)', 'rb') as file:
        lr = pickle.load(file)
    lr_predictor = lr.predict(X_test)
    # for i in range(20):
    #     print(test.iloc[i]['reviewid'], test.iloc[i]['score'], predictor[i])
    print('Logistic regression =', lr.score(X_test, y_test))
    print('starting graph', time.asctime())


    # nb = MultinomialNB()
    # nb.fit(X_train, y_train)
    # print('Naive Bayes =', nb.score(X_test, y_test))


    labels = [i for i in range(11)]
    conf_mat = confusion_matrix(y_test, lr_predictor)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    print('end', time.asctime())

    # prob_dist = cl.prob_classify((df.iloc[14000]['content']))
    # print(prob_dist.max(), prob_dist.prob('pos'), prob_dist.prob('neg'))
    # print(cl.accuracy(small_test))

    # blob = TextBlob(df.iloc[4]['content'])
    # print(blob.sentiment)
def process(text):
    tokens = TextBlob(text).words
    tokens = [t.lemmatize() for t in tokens if t not in stop_words]
    #tokens = [re.sub(r'[^\P{P}-]+', '', t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2]
    if len(tokens) > 0:
        return ' '.join(tokens)
    else:
        return None

def predictor(text):
    with open('models/LogisticRegression (newton)', 'rb') as mfile, open('vectorizer(1)', 'rb') as vfile:
        lr = pickle.load(mfile)
        vectorizer = pickle.load(vfile)

    return lr.predict(vectorizer.transform([process(text)]))

def artist_info(query):
    conn = sqlite3.connect('pitchfork.sqlite')
    c = conn.cursor()
    name = (f'%{query}%', )
    artist_list = []
    for row in c.execute('select distinct(artist) from artists where artist like ?', name):
        artist_list.append(row[0].title())
    conn.close()
    return artist_list

def get_artist(name):
    conn = sqlite3.connect('pitchfork.sqlite')
    df = pd.read_sql_query('select r.reviewid, r.title, r.score, r.pub_year, g.genre from reviews r, genres g where artist = ? and r.reviewid = g.reviewid', params=(name.lower(), ), con=conn)
    df_grouped = df.groupby(['title'])
    new_df = pd.DataFrame(columns=['reviewid', 'title', 'score', 'genre'])
    album_list = []
    for title, group in df_grouped:
        genre = []
        for index, row in group.iterrows():
            genre.append(row['genre'].title())
            score = row['score']
            year = row['pub_year']
            reviewid = row['reviewid']
        d = {'reviewid': reviewid, 'title': title.title(), 'score': score, 'year': year, 'genre': genre}
        album_list.append(d)
        new_df = new_df.append(d, ignore_index=True)
    new_df['year'] = new_df['year'].apply(int)
    new_df = new_df.sort_values('year', axis=0).reset_index(drop=True)
    album_list.sort(key=lambda x: x['year'])
    graph = create_graph(new_df)
    conn.close()
    return album_list, graph

def create_graph(df):
    df['title'] = df['title'].apply(lambda x: x[:15] + '...' if len(x) >= 15 else x)
    fig, ax = plt.subplots()
    sns.lineplot(df['title'], df['score'])

    #ax.tick_params(axis='x', labelsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

def review_content(id):
    conn = sqlite3.connect('pitchfork.sqlite')
    c = conn.cursor()
    reviewid = (id, )
    c.execute('select content from content where reviewid = ?', reviewid)
    content = c.fetchone()[0]
    conn.close()
    return content


if __name__ == '__main__':
    #blobbing()
    #artist_info('queen')
    #get_artist('queens of the stone age')
    review_content(22703)