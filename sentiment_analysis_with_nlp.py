#importing required libraries for the program 
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# reading required dataset
df=pd.read_csv('Dataset.csv')

# removing rows with missing reviews and keeping only the values with sentiment
df=df[['Review','Sentiment']].dropna()
df=df[df['Sentiment'].isin(['positive','negative'])]

# function to clean the data
def clean_data(review):
  review=review.lower()
  review=review.translate(str.maketrans('','',string.punctuation)) #to remove punctuation
  review=re.sub(r'\d+','',review)
  words=review.split()
  stop_words=text.ENGLISH_STOP_WORDS
  words=[word for word in words if word not in stop_words]
  return " ".join(words)

# applying the cleaning function to the data in the dataset
df['cleaned_data']=df['Review'].apply(clean_data)

# splitting data for input and output variables
x=df['cleaned_data']
y=df['Sentiment']

# splitting model for model evaluation
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# using tfidf vectorizer converting text to numeric
vector=TfidfVectorizer(ngram_range=(1,2),max_features=5000)
x_train_tfidf=vector.fit_transform(x_train)
x_test_tfidf=vector.transform(x_test)

# using losgistic regression model traininig the model
model=LogisticRegression()
model.fit(x_train_tfidf,y_train)

# make predictions
y_pred=model.predict(x_test_tfidf)

# preforming accuracy_score, classification report using model metrics form sklearn
print("Accuracy score: ",accuracy_score(y_test,y_pred))
print("Classification report of model is: ",classification_report(y_test,y_pred))

# data visualization using confusion matrix for better understanding
cm=confusion_matrix(y_test,y_pred, labels=['positive','negative'])
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,cmap='Blues',xticklabels=['positive','negative'],yticklabels=['positive','negative'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
