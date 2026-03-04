import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import re
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')





# Define stop words in Spanish using sklearn stop words and add a few custom ones
spanish_stop_words = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su",
    "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy",
    "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué",
    "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual",
    "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus",
    "ellas",
    "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo",
    "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros",
    "vuestras",
    "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis",
    "estén"
}
stop_words = set(stopwords.words('spanish')) | spanish_stop_words

# Initialize Spanish stemmer (alternative to lemmatization)
stemmer = SnowballStemmer("spanish")


def preprocess_text_alternative(text):
    # Normalize text: lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-záéíóúüñ ]', '', text)
    # Tokenize, remove stop words, and apply stemming
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(stemmed_tokens)



def classify_new_phrase_alternative(phrase,vectorizer_ngram,knn_preprocessed_alternative):
    # Preprocess the new phrase
    preprocessed_phrase = preprocess_text_alternative(phrase)

    # Transform the preprocessed phrase using the trained TF-IDF vectorizer with n-grams
    X_new_preprocessed_alternative = vectorizer_ngram.transform([preprocessed_phrase])

    # Predict the cluster for the new phrase
    predicted_cluster = knn_preprocessed_alternative.predict(X_new_preprocessed_alternative)[0]

    # Calculate the distance to the neighbors (for confidence score estimation)
    distances, indices = knn_preprocessed_alternative.kneighbors(X_new_preprocessed_alternative)

    # Compute a confidence score based on distances (inverse of the mean distance to neighbors)
    confidence_score = 1 / (np.mean(distances) + 1e-5)

    return predicted_cluster, confidence_score





# Define a function to apply classification to a DataFrame row and add predicted cluster and confidence score
def add_classification_to_row(row,vectorizer_ngram,knn_preprocessed_alternative):
    #print(row)
    predicted_cluster, confidence_score = classify_new_phrase_alternative(row['text'],vectorizer_ngram,knn_preprocessed_alternative)
    row['predicted_cluster'] = predicted_cluster
    row['confidence_score'] = confidence_score
    return row



# Define a function that, given a topic, returns the confidence score of the text for that topic
def get_confidence_score_for_topic(phrase, topic,vectorizer_ngram,knn_preprocessed_alternative):
    # Preprocess the new phrase
    preprocessed_phrase = preprocess_text_alternative(phrase)

    # Transform the preprocessed phrase using the trained TF-IDF vectorizer with n-grams
    X_new_preprocessed_alternative = vectorizer_ngram.transform([preprocessed_phrase])

    # Calculate the distance to the neighbors (for confidence score estimation)
    distances, indices = knn_preprocessed_alternative.kneighbors(X_new_preprocessed_alternative)

    # Check if the predicted cluster matches the given topic
    predicted_cluster = knn_preprocessed_alternative.predict(X_new_preprocessed_alternative)[0]
    if predicted_cluster == topic:
        # Compute a confidence score based on distances (inverse of the mean distance to neighbors)
        confidence_score = 1 / (np.mean(distances) + 1e-5)
    else:
        confidence_score = 0.0

    return confidence_score


def preprocess_text_alternative(text):
    # Normalize text: lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-záéíóúüñ ]', '', text)
    # Tokenize, remove stop words, and apply stemming
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(stemmed_tokens)