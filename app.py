from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

# Initialize flask application and porterstemmer
app = Flask(__name__)
porterStemmer = PorterStemmer()

# Load trained model and Tfidf Vectorizer
trainedModel = pickle.load(open('model2.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Define route for home page
@app.route('/', methods=['GET'])
def home_page():
    # Render the index.html template
    return render_template('index.html')

# Define function to predict the label of text
def predict(text):
    # Preprocess the text
    textReview = re.sub('[^a-zA-Z]', ' ', text)
    textReview = textReview.lower()
    textReview = textReview.split()
    textReview = [porterStemmer.stem(word) for word in textReview if not word in stopwords.words('english')]
    textReview = ' '.join(textReview)
    # Transform the preprocessed text into TFidf features
    review_vect = tfidfvect.transform([textReview]).toarray()
    # Make prediction using the loaded model
    prediction = 'FAKE' if trainedModel.predict(review_vect) == 0 else 'REAL'
    return prediction

# Define route for handling form submission
@app.route('/', methods=['POST'])
def webapp():
    # Get the text from the form
    text = request.form['text']
    # Make prediction
    prediction = predict(text)
    # Render the index.html template with the prediction result
    return render_template('index.html', text=text, result=prediction)

# Define route for API endpoint
@app.route('/predict/', methods=['GET','POST'])
def api():
    # Get the text from the query parameter
    text = request.args.get("text")
    # Make prediction
    prediction = predict(text)
    # Return prediction as JSON response
    return jsonify(prediction=prediction)


# Run the Flask application
if __name__ == "__main__":
    app.run()