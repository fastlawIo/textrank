#!/usr/bin/env python3

from flask import Flask, render_template, request, Markup
import markdown
from numpy import argsort
from textrank import summorize


app = Flask(__name__)

def to_html(text: str) -> str:
    r, txt = summorize(text)
    step = 100. / len(r)
    res = []
    for i, t in zip(r, txt):
        res.append(f'<span data-val={round(i * step)}>{t}</span>')
    return ' '.join(res)

app = Flask(__name__)

@app.route('/summorize', methods=['POST'])
def summorize_pg():
    txt = request.form.get('text')
    html = to_html(txt)
    return render_template('./index.html', html=Markup(html))

@app.route('/index', methods=['GET'])
def index():
    return render_template('./index.html')

if __name__ == '__main__':
    app.run(port=8080)
