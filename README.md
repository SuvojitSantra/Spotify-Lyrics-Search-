# Spotify Lyrics Search

A Python-based recommendation system that finds songs based on lyric snippets using TF-IDF vectorization and Cosine Similarity. This project uses the "Spotify Million Song Dataset" to analyze and match user input with song lyrics.

## Features

-   **Lyric Search**: Input a phrase or sentence from a song, and the system finds the closest matching song.
-   **Text Preprocessing**: Uses NLTK for tokenization, lemmatization, and stop-word removal to improve search accuracy.
-   **TF-IDF Vectorization**: Converts lyrics into numerical vectors to measure importance.
-   **Cosine Similarity**: Calculates the similarity between the user's query and the dataset to find the best match.

## Dataset

The project relies on the `Spotify Million Song Dataset_exported.csv`. Ensure this file is present in the project directory.
The CSV should contain at least the following columns:
-   `artist`
-   `song` (renamed to `track_name` in the script)
-   `text` (renamed to `lyrics` in the script)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SuvojitSantra/Spotify-Lyrics-Search-.git
    cd Spotify-Lyrics-Search-
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1.  Load the dataset.
2.  Preprocess the lyrics (this may take a moment).
3.  Build the TF-IDF model.
4.  Run a sample query ("look at her face it's a wonderful face") and print the result.

## Algorithm Details

1.  **Preprocessing**: Lowercasing, removing non-alphabetic characters, and lemmatizing words.
2.  **Vectorization**: `TfidfVectorizer` from Scikit-Learn is used with a max feature size of 5000 and unigram+bigram range.
3.  **Matching**: Cosine similarity scores are computed between the query vector and all song vectors. The highest score determines the best match.

## License

This project is open-source.
