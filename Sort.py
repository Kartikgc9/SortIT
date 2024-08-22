import os
import googleapiclient.discovery
import httplib2
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import tkinter as tk
from googleapiclient.http import MediaFileUpload, HttpRequest
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model = load_model('C:/path/to/sentiment_analysis_model.h5')

http = httplib2.Http(timeout=30)

def retry_func(retry_state):
    if retry_state.num_retries > 3:
        return None
    else:
        return 1  # wait 1 second before retrying

HttpRequest.retry_func = retry_func

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Set up API credentials
api_key = "AIzaSyBzlLBN2L_su07FO0G_Gpjo5EskbX37qYE"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# Set up video ID
video_id = "MfdXFbB-1oo"

# Fetch comments
comments = []
next_page_token = ""
while True:
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        pageToken=next_page_token
    )
    response = request.execute()
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]
        text = comment["snippet"]["textDisplay"]
        comments.append(text)
    next_page_token = response.get("nextPageToken")
    if not next_page_token:
        break

# Load the trained CNN model
model = load_model('sentiment_analysis_model.h5')

# Create a tokenizer to preprocess the comments
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(comments)

# Preprocess the comments
sequences = tokenizer.texts_to_sequences(comments)
padded_sequences = pad_sequences(sequences, maxlen=200)

# Make predictions using the CNN model
predictions = model.predict(padded_sequences)

# Get the sentiment labels with the highest probability
sentiment_labels = np.argmax(predictions, axis=1)

# Map the sentiment labels to sentiment categories (0: negative, 1: neutral, 2: positive)
sentiment_categories = ['Negative', 'Neutral', 'Positive']
categories = {}
for i, sentiment_label in enumerate(sentiment_labels):
    category = sentiment_categories[sentiment_label]
    if category not in categories:
        categories[category] = []
    categories[category].append(comments[i])

# Calculate most common category
most_common_category = max(categories, key=lambda x: len(categories[x]))

# Write output to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Most common category: {}\n".format(most_common_category))
    f.write("Comments in this category:\n")
    for comment in categories[most_common_category]:
        f.write("{}\n".format(comment))
    f.write("Other categories:\n")
    for category, comments in categories.items():
        if category != most_common_category:
            f.write("{}: {} comments\n".format(category, len(comments)))
    f.write("Top 3 words in each category:\n")
    for category, comments in categories.items():
        translator = str.maketrans('', '', string.punctuation + '’“”')
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = []
        for comment in comments:
            comment_no_punct = re.sub(r'\W+', ' ', comment).translate(translator)
            tokens = word_tokenize(comment_no_punct)
            tokens = [word for word in tokens if word.isalpha() and word.lower() not in stop_words]
            words.extend(tokens)
        top_words = Counter(words).most_common(3)
        f.write("{} top words: {}\n".format(category, top_words))

# Display output in GUI window
root = tk.Tk()
root.title("Sentiment Analysis")

text_widget = tk.Text(root)
text_widget.pack()

text_widget.insert(tk.INSERT, "Most common category: {}\n".format(most_common_category))
text_widget.insert(tk.INSERT, "Comments in this category:\n")
for comment in categories[most_common_category]:
    text_widget.insert(tk.INSERT, "{}\n".format(comment))
text_widget.insert(tk.INSERT, "Other categories:\n")
for category, comments in categories.items():
    if category != most_common_category:
        text_widget.insert(tk.INSERT, "{}: {} comments\n".format(category, len(comments)))
text_widget.insert(tk.INSERT, "Top 3 words in each category:\n")