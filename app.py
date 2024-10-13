import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas as pd

app=Flask(__name__)
global Classifier
global Vectorizer

data = pd.read_csv('/workspaces/Spam-Or-Ham/spam (1).csv', encoding='latin-1')


Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(data.v2)
Classifier.fit(vectorize_text, data.v1)

@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message','')
    error = ''
    prediction_probability = ''
    prediction = ''

    try:
        if(len(message) > 0):
            vectorize_message = Vectorizer.transform([message])
            prediction = Classifier.predict(vectorize_message)[0]
            prediction_probability = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__)+ ' ' + str(inst)
    return jsonify(
        message = message, prediction_probability = prediction_probability, 
        prediction = prediction, error = error
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)


#curl "http://localhost:5000/?message=your_message"
