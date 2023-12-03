# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# %%
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# %%
data = pd.read_csv("fake_or_real_news.csv")

# %%
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing special characters
    text = re.sub(r'\W', ' ', text)

    # # Tokenization
    # words = word_tokenize(text)

    # # Removing Stop Words and Applying Stemming
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    # # Joining Words
    # text = ' '.join(words)
    
    return text


# %%
# Assuming 'data' is your DataFrame and 'text' is the column with news articles
# data['text'] = data['text'].apply(preprocess_text)


# %%
data

# %%
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)

# %%
X, y = data["text"], data["fake"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% [markdown]
# TF-IDF: Term Frequency - Inverse Document Frequency
# 
# A metric that indicates how important a word is to a document in a collection. It weighs the improtance of each word in a document based on how often it appears in that document and how often it appears accross all documents in the collection.
# 
# TF: Number of times a term t appears in a document
# IDF: Logarithm of total number of documents divided by no. of docs that contain term
# TF-IDF: TF * IDF
# 
# Basically allows us to find the most relevant and distinctive words per document.

# %%
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
vectorizer = TfidfVectorizer(stop_words='english', 
                             max_df=0.7,  # terms that appear in more than 70% of documents are ignored
                             min_df=3,    # terms that appear in less than 3 documents are ignored
                             ngram_range=(1, 3))  # unigrams and bigrams are considered
X_train_vectorized = vectorizer.fit_transform(X_train) 
X_test_vectorized = vectorizer.transform(X_test)

# %%
clf = LinearSVC()  # Linear SVC is considered one of the best text classification algorithms
clf.fit(X_train_vectorized, y_train)
clf.score(X_test_vectorized, y_test)

# %%
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
clf.score(X_test_vectorized, y_test)


# %%


# %%
article_text = X_test.iloc[10]
vectorized_text = vectorizer.transform([article_text])

# %%
clf.predict(vectorized_text)

# %%
y_test.iloc[10]

# %%
# Generate a random index within the range of X_test
random_index = random.randint(0, len(X_test) - 1)

# Select a random article using the generated index
article_text = X_test.iloc[random_index]
vectorized_text = vectorizer.transform([article_text])

# Predict
predicted_value = clf.predict(vectorized_text)[0]

# Actual value
actual_value = y_test.iloc[random_index]

# Comparison and output
if predicted_value == actual_value:
    print("The answer is correct.")
else:
    print("The answer is incorrect.")



# %%
# Predict for the entire test set
predictions = clf.predict(X_test_vectorized)

# Convert predictions and actual labels to arrays for easier comparison
predictions_array = np.array(predictions)
actual_labels_array = y_test.to_numpy()

# Find indices where predictions and actual labels differ
mismatch_indices = np.where(predictions_array != actual_labels_array)[0]

# Print each mismatch with its index
for index in mismatch_indices:
    print(f"Index: {index}, Predicted: {predictions_array[index]}, Actual: {actual_labels_array[index]}")
    print(f"Text: {X_test.iloc[index]}\n")


# %%
# Predict for the entire test set
predictions = clf.predict(X_test_vectorized)

# Convert predictions and actual labels to arrays for easier comparison
predictions_array = np.array(predictions)
actual_labels_array = y_test.to_numpy()

# Initialize lists for false negatives and false positives
false_negatives = []
false_positives = []

# Iterate over the predictions and actual labels
for i in range(len(predictions_array)):
    if predictions_array[i] == 0 and actual_labels_array[i] == 1:
        # False Negative
        false_negatives.append(i)
    elif predictions_array[i] == 1 and actual_labels_array[i] == 0:
        # False Positive
        false_positives.append(i)

# %%
# Print False Negatives
print("False Negatives: ACTUALLY FAKE")
for index in false_negatives:
    print(f"Index: {index}, Text: {X_test.iloc[index]}")
    print()
    print()
    print()
    

# %%
# Print False Positives
print("\nFalse Positives: ACTUALLY REAL")
for index in false_positives:
    print(f"Index: {index}, Text: {X_test.iloc[index]}")
    print()
    print()
    print()

# %%
# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plotting using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()



