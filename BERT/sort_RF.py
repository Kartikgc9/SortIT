import googleapiclient.discovery
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import ttk, scrolledtext

# NLTK downloads
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
import re
# Function to remove emojis using regex
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002500-\U00002BEF"  # Chinese/Japanese/Korean characters
        "\U00002702-\U000027B0"  # Dingbats
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001f926-\U0001f937"  # Supplemental symbols and pictographs
        "\U00010000-\U0010ffff"  # Supplementary Private Use Area-A
        "\u200d"  # Zero width joiner
        "\u2640-\u2642"  # Gender symbols
        "\u2600-\u2B55"  # Miscellaneous symbols
        "\u23cf"  # Additional transport and map symbols
        "\u23e9"  # Additional symbols
        "\u231a"  # Watch and clock symbols
        "\ufe0f"  # Variation selector
        "\u3030"  # Wavy dash
        "]+",
        re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Function to detect spam (basic heuristic example)
def is_spam(comment):
    spam_keywords = ["buy", "free", "win", "prize", "click here", "subscribe"]
    return any(keyword in comment.lower() for keyword in spam_keywords)

# Set up API credentials
api_key = "AIzaSyBzlLBN2L_su07FO0G_Gpjo5EskbX37qYE"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# Set up video ID
video_id = "s3XbWJJYfmQ"

# Fetch comments
comments_list = []
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
        # Remove emojis and filter spam comments
        cleaned_text = remove_emojis(text)
        if not is_spam(cleaned_text):
            comments_list.append(cleaned_text)
    next_page_token = response.get("nextPageToken")
    if not next_page_token:
        break

# Generate sentiment labels using NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sentiment_labels = []
for comment in comments_list:
    score = sia.polarity_scores(comment)['compound']
    if score >= 0.05:
        sentiment_labels.append(2)  # Positive
    elif score <= -0.05:
        sentiment_labels.append(0)  # Negative
    else:
        sentiment_labels.append(1)  # Neutral

# Preprocess comments with TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(3, 4))
tfidf_vectors = vectorizer.fit_transform(comments_list)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=50)
rf_model.fit(tfidf_vectors, sentiment_labels)

# Make predictions using the Random Forest model
predictions = rf_model.predict(tfidf_vectors)

# Sentiment categories
sentiment_categories = ['Negative', 'Neutral', 'Positive']
categories = {category: [] for category in sentiment_categories}

for i, sentiment_label in enumerate(predictions):
    category = sentiment_categories[sentiment_label]
    categories[category].append(comments_list[i])

# Function to get top 3 words in each category
def get_top_words(comments):
    translator = str.maketrans('', '', string.punctuation + '’“”')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = []
    for comment in comments:
        comment_no_punct = re.sub(r'\W+', ' ', comment).translate(translator)
        tokens = word_tokenize(comment_no_punct)
        tokens = [word for word in tokens if
                  word.isalpha() and word.lower() not in stop_words and len(word) > 2 and not word.isdigit()]
        words.extend(tokens)
    top_words = Counter(words).most_common(3)
    return top_words

top_words_by_category = {category: get_top_words(comments) for category, comments in categories.items()}

# GUI setup
root = tk.Tk()
root.title("Sortit")

# Create tabs for each sentiment category
tab_control = ttk.Notebook(root)
tabs = {}

for category in sentiment_categories:
    tab = ttk.Frame(tab_control)
    tab_control.add(tab, text=category)
    tabs[category] = tab

tab_control.pack(expand=1, fill="both")

# Display comments and top words in each tab
for category, tab in tabs.items():
    # Scrollable text box for comments
    txt = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=100, height=20)
    txt.pack(expand=1, fill="both")

    txt.insert(tk.END, f"Top 3 words: {top_words_by_category[category]}\n\n")
    for comment in categories[category]:
        txt.insert(tk.END, comment + "\n\n")

# Start the GUI event loop
root.mainloop()