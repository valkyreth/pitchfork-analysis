from flask import Flask, render_template, redirect, request, url_for, flash
from sentiment import predictor, artist_info, get_artist, review_content
import json

app = Flask(__name__)
app.secret_key = "key_pitchfork"


@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == "__main__":
    app.run(debug=True)
