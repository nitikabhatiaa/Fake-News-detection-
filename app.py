from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    input_vector = tfidf.transform([input_text]).toarray()
    prediction = model.predict(input_vector)
    result = "Real News" if prediction[0] == 1 else "Fake News"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
