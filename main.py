import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def download_nltk_data():
    """Download necessary NLTK data if not present."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def load_and_preprocess_data(filepath):
    """Load dataset and preprocess lyrics."""
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return None, None
        
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")

    # Rename columns to standardized names
    # Note: Adjust these if your CSV has different column names
    if 'song' in df.columns:
        df = df.rename(columns={"song": "track_name", "text": "lyrics"})
    
    # Select relevant columns
    required_cols = ['artist', 'track_name', 'lyrics']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        print(f"Found: {df.columns.tolist()}")
        return None, None

    df = df[required_cols]
    df.dropna(inplace=True)
    
    print(f"Cleaned dataset shape: {df.shape}")
    
    # Preprocessing setup
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    print("Preprocessing lyrics (this may take a while)...")
    df['clean_lyrics'] = df['lyrics'].apply(preprocess)
    
    return df, lemmatizer, stop_words # returning helpers if needed outside

def build_vectorizer(df):
    """Build TF-IDF Vectorizer."""
    print("Building TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean_lyrics'])
    print(f"tfidf matrix shape: {X.shape}")
    return vectorizer, X

def predict_song(lyric_snippet, vectorizer, X, df, lemmatizer, stop_words):
    """Find best matching song for a lyric snippet."""
    def preprocess_query(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    cleaned_snippet = preprocess_query(lyric_snippet)
    snippet_vector = vectorizer.transform([cleaned_snippet])

    similarity_scores = cosine_similarity(snippet_vector, X)
    best_match_index = similarity_scores.argmax()

    return {
        "Song Title": df.iloc[best_match_index]['track_name'],
        "Artist": df.iloc[best_match_index]['artist'],
        "Confidence": similarity_scores[0][best_match_index]
    }

def main():
    download_nltk_data()
    
    csv_file = "Spotify Million Song Dataset_exported.csv"
    df, lemmatizer, stop_words = load_and_preprocess_data(csv_file)
    
    if df is None:
        return

    vectorizer, X = build_vectorizer(df)
    
    # Example Query
    query = "look at her face it's a wonderful face"
    print(f"\nSearching for lyrics: '{query}'")
    result = predict_song(query, vectorizer, X, df, lemmatizer, stop_words)
    
    print("\nObservation Result:")
    print(result)
    
    # You can add the evaluation logic here if needed, but for a run script, a single prediction is often enough demo.

if __name__ == "__main__":
    main()
