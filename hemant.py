import pandas as pd
#import database_connection as db
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.model_selection import cross_val_score
import sqlite3

conn = sqlite3.connect('pitchfork.sqlite')
data = pd.read_sql_query("select r.score, c.content from reviews r, content c where r.reviewid = c.reviewid ", conn)
data.to_pickle("./reviews.pkl")
#db.close_db(c, conn)
conn.close()

data = pd.read_pickle("./reviews.pkl")
data['rounded'] = [round(score) for score in data.score]

# Data distribution
# fig = plt.figure(figsize=(10, 10))
# data.groupby('rounded').content.count().plot.bar(ylim=0)
# plt.show()

data = data[pd.notnull(data['content'])]

x_train, x_test, y_train, y_test = train_test_split(data['content'], data['rounded'], random_state=0)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words="english")


X_train = tfidf.fit_transform(x_train.tolist())
X_test = tfidf.transform(x_test.tolist())
y_train = y_train.tolist()
y_test = y_test.tolist()

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]


def training(x, y, model):
    lr = model
    lr.fit(x, y)
    with open(model.__class__.__name__, 'wb') as file:
        pickle.dump(lr, file)
    # with open(model.__class__.__name__, 'wb') as file:
    #     lr = pickle.load(file)
    return lr


for i in models:
    model = training(X_train, y_train, i)
    print(i.__class__.__name__ + " : " + str(model.score(X_test, y_test)))


CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
