import sqlite3
#import pandas as pd
from pandas import DataFrame
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
conn = sqlite3.connect('pitchfork.sqlite')
c = conn.cursor()
sns.set(style="white", context="talk")
rs = np.random.RandomState(8)

# Set up the matplotlib figure

#f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)


def read_from_db():
    #pan = c.execute('select artist,avg(score) from reviews group by artist order by avg(score) desc')
    #pan1 = c.execute('select reviewid, genre from genres group by genre ')
    #pan2 = c.execute('select artist, avg(score) from reviews group by artist having count(reviewid) > 4 order by avg(score) desc')
    #print(pan1.fetchall())
    #pan = c.execute('select * from reviews')
    #print(type(pan))
    #df = DataFrame(pan.fetchall())
    #df.columns = ['reviewid','title','artist','url','score','best_new_music','author','author_type','pub_date','pub_weekday','pub_day','pub_month','pub_year']
    #print(df.dtypes)
    '''df1 = df.groupby('artist').agg({'score':np.mean}).sort_values(by='score', ascending=False)
    df2 = df1.head(10)
    df3 = df1.tail(10)
    df4 = pd.concat([df2,df3])'''
    #data = c.fetchall()
    #print(df4)
    #plt.plot(arti)
    #print(df.head(10))
    #print(type(df))
    #print(data[0][1])

    #print("reviewid")
    #for row in c.fetchall():
        #print(row)
        #if(row[4]>9):
            #print(row[4])'''
    #for row1 in pan2.fetchall():
     #   print(row1)
    #g = plt.bar(pan2.tail(10)['artist'], pan2.tail(10)['avg(score)'])
    #h = DataFrame(pan2.fetchall())
    #g = h.head(10)
    #plt.bar(g.artist, g. avg(score))
    #plt.xlabel('artist', fontsize=5)
    #plt.ylabel('avg(score)', fontsize=5)
    #plt.xticks(index, label, fontsize=5, rotation=30)
    #plt.show()

    q1 = c.execute('select r.pub_year,g.genre,avg(r.score) from genres g, reviews r where g.reviewid = r.reviewid group by g.genre,r.pub_year order by r.pub_year,avg(r.score) desc')
    #print(q1.fetchall())
    df1 = DataFrame(q1.fetchall())
    #print(df1)
    q2 = c.execute('select g.genre,avg(r.score) from genres g, reviews r where g.reviewid = r.reviewid group by g.genre order by avg(r.score) desc')
    df2 = DataFrame(q2.fetchall())
    df2 = df2.drop(index=4)
    l = df2[0].tolist()
    y_pos = [i for i, _ in enumerate(l)]
    m = df2[1].tolist()

    g = sns.barplot(m, l, data=df2)
    #plt.xlim(6.5, 7.5)
    r = np.linspace(0, 9, 10)
    j = 0

    for row in df2.itertuples():
        g.text(x=row[2]+0.2, y=j, s='{:4.2f}'.format(row[2]), color='black', ha='center')
        j += 1


    #plt.bar(m,y_pos)
    #plt.yticks(y_pos, l)
    #plt.legend()
    #plt.xlabel('bar number')
    #plt.ylabel('bar height')

    #plt.title('Epic Graph\nAnother Line! Whoa')
    plt.tight_layout()
    plt.show()

read_from_db()
c.close()
conn.close()