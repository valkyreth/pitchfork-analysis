import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(style="darkgrid")
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)


conn = sqlite3.connect('pitchfork.sqlite')
c = conn.cursor()

def artists_score():
    data = pd.read_sql_query("select artist, avg(score) from reviews group by artist having count(reviewid) > 4 order by avg(score) desc", conn)
    #data1 = pd.read_sql_query("select avg(score) from reviews where artist like '2 chainz'", conn)
    print(data.shape)
    g = sns.barplot(data.head(10)['artist'], data.head(10)['avg(score)'], data=data.head(10))
    #sbn.barplot(data=data.head())
    #g = plt.bar(data.tail(10)['artist'], data.tail(10)['avg(score)'])
    plt.tight_layout()
    plt.xticks(rotation=30)

    r = np.linspace(0,9,10)
    print(r)
    j = 0

    for row in data.head(10).itertuples():
        g.text(x=r[j], y=row[2], s='{:4.2f}'.format(row[2]), color='black', ha='center')

        j += 1

def weekday_score():
    data = pd.read_sql_query("select pub_weekday, avg(score) from reviews group by pub_weekday order by pub_weekday asc", conn)

    g = sns.barplot(data['pub_weekday'], data['avg(score)'], data=data)
    g.set(xticklabels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    r = np.linspace(0,6,7)
    print(r)
    j = 0

    for row in data.itertuples():
        g.text(x=r[j], y=row[2], s='{:4.2f}'.format(row[2]), color='black', ha='center')
        print(row)
        j += 1

def genre_change():
    data = pd.read_sql_query("select genres.reviewid, genres.genre, reviews.pub_year from genres inner join reviews on genres.reviewid=reviews.reviewid", conn)
    #g = sbn.barplot(data['pub_weekday'], data['avg(score)'], data=data)
    #g.set(xticklabels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    data = data.dropna(axis=0,how='any')
    print(data.shape)
    fig, ax = plt.subplots(figsize=(15,7))
    data.groupby(by=['pub_year', 'genre']).count()['reviewid'].unstack().plot(ax=ax)
    plt.xticks(np.arange(2000,2020,step=5))
    nd = data.groupby(by=['pub_year', 'genre']).count()['reviewid'].unstack()
    print(nd['rock'].max())

    #sns.relplot(x=nd['pub_year'],y=nd['genre'], data=nd)
    #print(grouped.describe())
    #nd = grouped.count()

    l = []
    # for key1,key2  in nd:
    #     max = 0
    #     val = nd.get_group((key1,key2))
    #     if val > max:
    #         max = val
    #     print(grouped.get_group(key), "\n\n")

    r = np.linspace(0,18,19)

    j = 0

    for row in data.head().itertuples():
        #g.text(x=r[j], y=row[2], s=row[2] color='black', ha='center')
        #print(row)
        j += 1

artists_score()
plt.tight_layout()
plt.show()
conn.close()



