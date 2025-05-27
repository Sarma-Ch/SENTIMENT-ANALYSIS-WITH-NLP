# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: CHAVALI NARASIMHA SARMA

INTERN ID: CT06DM1358

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTHOSH KUMAR

DESCRIPTION:

Sentiment Analysis is one of the most important and commonly used applications in Natural Language Processing (NLP). It is the process of identifying and categorizing opinions expressed in a piece of text,

especially to determine whether the writer’s attitude toward a particular topic, product, or service is positive, negative, or neutral. This task is widely used in industries such as e-commerce, marketing,

customer service, and social media analysis to understand how people feel about products, services, or events.

Sentiment analysis helps businesses by giving them the ability to analyze large volumes of customer feedback automatically, without the need for manual review. This allows companies to quickly respond to customer

complaints, improve products, and make better decisions based on user experiences. For example, brands can monitor public opinion on social media, researchers can follow political trends or survey results, and

developers can create tools for automated review analysis.

For example 'I love this Food' is labelled as Positive , 'The movie was terrible' is labelled as Negative.

This program uses Term frequency-inverse document frequency vectorizer(TF-IDF VECTORIZER) and Logistic Regression linear model for sentiment Analysis using NLP

code implementation description includes:

Importing Libraries: Pandas used for loading and managing dataset.

re and string libraries are used for cleaning and formatting the text from dataset.

Scikit learn is a advanced machine learning library used for model selection model building evaluation and initializing Vectorizer.

matplotlib.pyplot and seaborn are used for plotting and visualization of model using confusion matrix.

Loading and preparing the dataset: Using pandas.read_csv() method used to read the dataset which is in the form of comma seperated value.

We only kept the Reviews and Sentiment columns and .dropna() method is used to remove the rows that has missing values.

Text Preprocessing: We defined a custom function named clean_data to clean the dataset for model evaluation.

This function first converts all words to lower case usig lower() and then removes the punctuation in the dataset using string library.

Then removes all the digits and also removes stop_words from the dataset and this function is called to clean the data.

Splitting data: Using train_test_split() from sklearn.model_selction we split the data and here in general 80% will be training data and 20% will be testing data.

TF-IDF Vectorization: Machine learning models cannot work with text data. So we convert the data of text into numeric using TF-IDF Vectorization.And we use bigrams to know how the words are consequent words are related to each other

The mathematical formula for calculating tf-idf vector is

[TF-IDFi=TFi*ln(1+(N/Ni))] where [TFi=(Nunber of occurances of i in document)/(Total number of words in document)] and Ni=Number of documents that contain word i , N=Total number of documents in the dataset.

Model Building: Logistic Regression is the important Linear model that helps in model evaluation of binary tasks

Evaluaiting Model: Accuracy Score: Tells us how many predictions were correct.

Classification Report: Shows precision, recall, and F1-score for both positive and negative classes.

Confusion Matrix: A visual way to understand how many true/false positives and negatives occurred.

Dataset used for Sentiment Analysis using NLP is: The dataset used for this program was downloaded from Kaggle. It contains 205,052 entries from Flipkart, an Indian e-commerce platform. Each entry consists of a customer review and a labeled sentiment, which can

be positive, negative, or neutral. This real-world dataset provides a great opportunity to build and test a practical sentiment analysis model using natural language processing techniques.

** Output **


