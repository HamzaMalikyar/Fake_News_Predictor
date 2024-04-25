import pandas as pd

# Load true dataset
true = pd.read_csv('True.csv')
# Visualize true dataset
print(true.head(3))
print(true.shape)
# Load fake dataset and visualize fake dataset
fake = pd.read_csv('Fake.csv')
print(fake.shape)

# Add label columns
true['label'] = 1
fake['label'] = 0

# Concatenate first 5000 samples of both datasets
samples = [true.loc[:5000][:], fake.loc[:5000][:]]
df = pd.concat(samples)

# Visualize concatenated data
print(df.shape)
print(df.tail())

X = df.drop('label', axis=1)
y = df['label']

# Drop rows with missing values and create copy of clean data
df = df.dropna()
df2 = df.copy()

# Visualize clean data
print(df2.head())

# Reset index of copied data and visualize
df2.reset_index(inplace=True)
print(df2.head())

# Import necessary libraries for preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
porterStemmer = PorterStemmer()

import re
import nltk

# Download stopwords from NLTK
nltk.download('stopwords')

# Empty list to store preprocessed text
corpus = []
# Iterate over each row in copied dataset
for i in range(0, len(df2)):
    # Remove non-alphabetic characters and convert to lowercase
    textReview = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
    textReview = textReview.lower()
    textReview = textReview.split()
    # Remove stopwords and perform stemming
    textReview = [porterStemmer.stem(word) for word in textReview if not word in stopwords.words('english')]
    textReview = ' '.join(textReview)
    corpus.append(textReview)

# Import TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize TFidf Vectorizer with max_features and ngram_range
tfidfVectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

# Convert the corpus into TFidf features
X = tfidfVectorizer.fit_transform(corpus).toarray()
y = df2['label']

# Split the dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

# Initialize classifier with max_iter = 1000 to prevent overfitting
pac = PassiveAggressiveClassifier(max_iter=1000)

from sklearn import metrics
import numpy as np
import itertools

# Fit classifier into training data
pac.fit(X_train, y_train)

# Make predictions on test data
pred = pac.predict(X_test)

# Calculate and predict accuracy score
accuracyScore = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % accuracyScore)

import matplotlib.pyplot as plt

# Visualize using confusion matrix
def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute and plot confusion matrix
cm = metrics.confusion_matrix(y_test, pred)
confusion_matrix(cm, classes=['FAKE', 'REAL'])

# Preprocess sample review for prediction
textReview = re.sub('[^a-zA-Z]', ' ', fake['text'][11110])
textReview = textReview.lower()
textReview = textReview.split()
textReview = [porterStemmer.stem(word) for word in textReview if not word in stopwords.words('english')]
textReview = ' '.join(textReview)
print(textReview)

# Transform preprocessed review using Tfidf Vectorizer
val = tfidfVectorizer.transform([textReview]).toarray()

# Save classifier and vectorizer to disk
import pickle

pickle.dump(pac, open('model2.pkl', 'wb'))
pickle.dump(tfidfVectorizer, open('tfidfvect2.pkl', 'wb'))

# Load saved trainedModel and vectorizer
joblib_model = pickle.load(open('model2.pkl', 'rb'))
joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Transform preprocessed review using loaded Tfidf Vectorizer
val_pkl = joblib_vect.transform([textReview]).toarray()

# Make predictions using loaded trainedModel
joblib_model.predict(val_pkl)