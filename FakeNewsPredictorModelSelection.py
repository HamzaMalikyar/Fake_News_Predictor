import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Read fake and true datasets
fakeData=pd.read_csv('Fake.csv')
trueData=pd.read_csv('True.csv')

# Add a 'class' column with label 0 for fake news and label 1 for true news
fakeData["class"]=0
trueData['class']=1

# Select last 10 rows from each dataset for manual testing and drop them from the original datasets
fakeDataManualTesting = fakeData.tail(10)
for i in range(23480, 23470, -1):
    fakeData.drop([i], axis=0, inplace=True)

trueDataManualTesting = trueData.tail(10)
for i in range(21416, 21406, -1):
    trueData.drop([i], axis=0, inplace=True)

# Add 'class' column to manual testing datasets
fakeDataManualTesting['class']=0
trueDataManualTesting['class']=1

# Merge datasets
mergeData=pd.concat([fakeData, trueData], axis = 0)

# Drop unnecessary columns
data=mergeData.drop(['title', 'subject', 'date'], axis = 1)

# Shuffle the data
data = data.sample(frac = 1)

# Reset index
data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)

# Text preprocessing function
def textPreprocessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

# Apply text preprocessing to 'text' column
data['text'] = data['text'].apply(textPreprocessing)

x = data['text']
y = data['class']

# Split data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25)

# Tfidf vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xvTrain = vectorization.fit_transform(xTrain)
xvTest = vectorization.transform(xTest)

# Logistic regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
print(LR.fit(xvTrain, yTrain))
logisticRegressionPrediction = LR.predict(xvTest)
print("%0.3f" %LR.score(xvTest, yTest))
print(classification_report(yTest, logisticRegressionPrediction))

# Passive aggressive classifier
from sklearn.linear_model import PassiveAggressiveClassifier
PAC = PassiveAggressiveClassifier()
print(PAC.fit(xvTrain, yTrain))
passiveAggressivePrediction = PAC.predict(xvTest)
print("%0.3f" %PAC.score(xvTest, yTest))
print(classification_report(yTest, passiveAggressivePrediction))

# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
print(DT.fit(xvTrain, yTrain))
decisionTreePrediction = DT.predict(xvTest)
print("%0.3f" %DT.score(xvTest, yTest))
print(classification_report(yTest, logisticRegressionPrediction))

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
print(RF.fit(xvTrain, yTrain))
randomForestPrediction = RF.predict(xvTest)
print("%0.3f" %RF.score(xvTest, yTest))
print(classification_report(yTest, randomForestPrediction))

# Function to convert numeric label to text label
def labelOutput(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

# Function for manual testing
def manualTesting(news):
    newsTest = {"text": [news]}
    newDefTest = pd.DataFrame(newsTest)
    newDefTest['text'] = newDefTest["text"].apply(textPreprocessing)
    newXTest = newDefTest["text"]
    newXVTest = vectorization.transform(newXTest)
    lrPrediction = LR.predict(newXVTest)
    dtPrediction = DT.predict(newXVTest)
    rfPrediction = RF.predict(newXVTest)
    paPrediction = PAC.predict(newXVTest)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction: {} \nPAC Prediction: {}".format(
        labelOutput(lrPrediction[0]),
        labelOutput(dtPrediction[0]),
        labelOutput(rfPrediction[0]),
        labelOutput(paPrediction[0])))

# Get input from user for manual testing
print("Please enter news you would like to check: ")
news = str(input())
manualTesting(news)