from flask import Flask, render_template, request

import tf_keras as keras
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

app = Flask(__name__)

# Load the emotion analysis model
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion_label = None
    if request.method == 'POST':
        sentence = request.form['sentence']
        emotion_labels = emotion(sentence)
        emotion_label = emotion_labels[0]['label']
    return render_template('index.html', emotion_label=emotion_label)

if __name__ == '__main__':
    # Set debug to False and configure host and port
    app.run(debug=False, host='0.0.0.0', port=8000)