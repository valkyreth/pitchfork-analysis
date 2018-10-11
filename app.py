from flask import Flask,render_template,redirect,request,url_for,session,g
app = Flask(__name__)
app.secret_key = "key_pitchfork"


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
