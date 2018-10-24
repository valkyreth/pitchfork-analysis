from flask import Flask, render_template, redirect, request, url_for, flash
from sentiment import predictor, artist_info, get_artist, review_content, get_score, conf_mat, get_albums, get_genres
import json
from graphs import *

app = Flask(__name__)
app.secret_key = "key_pitchfork"


@app.route('/')
def index():
    genres = get_genres()
    return render_template('index.html', genres=genres)

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        s = request.form['review']
        #print(predictor(s))
        res = str(predictor(s)[0])
        flash(f'The predicted score is {res}')
        return redirect(url_for('predict'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form['artist']
    artists = artist_info(query)
    return render_template('searchResults.html', artists=artists)

@app.route('/artist_stats', methods=['GET', 'POST'])
def artist_stats():
    name = request.form['name']
    albums, graph = get_artist(name)
    #display_artist(albums, graph)
    return render_template('artist.html', albums=albums, graph=graph)
    #return render_template('index.html')
    #return json.dumps({'status': 'OK'})

@app.route('/get_review', methods=['GET', 'POST'])
def get_review():
    id = request.get_json()['id']
    print(id, type(id))
    content = review_content(int(id))
    return json.dumps({'status': 'OK', 'data': content})

@app.route('/insights', methods=['GET', 'POST'])
def insights():
    graphs = Graph().all_graphs()
    return render_template('graphs.html', graphs=graphs)

@app.route('/models/<name>', methods=['GET', 'POST'])
def models(name):
    score = get_score(name)
    graph = conf_mat(name)
    return render_template(f'{name}.html', score=score, graph=graph)

@app.route('/genre_albums', methods=['GET', 'POST'])
def genre_albums():
    name = request.form['Genre']
    albums = get_albums(name)
    return render_template('albums.html', albums=albums)

if __name__ == "__main__":
    app.run(debug=True)
