"""Lexical similarity features."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


def calculate_lexical_similarity(text_a: str, text_b: str) -> float:
    """Calculate TF-IDF cosine similarity."""
    if not text_a or not text_b:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000
        )

        vectors = vectorizer.fit_transform([text_a, text_b])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        return float(similarity)
    except:
        return 0.0


def calculate_word_overlap(text_a: str, text_b: str) -> float:
    """Calculate word overlap ratio."""
    if not text_a or not text_b:
        return 0.0

    words_a = set(re.findall(r'\w+', text_a.lower()))
    words_b = set(re.findall(r'\w+', text_b.lower()))

    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)

    return intersection / union if union > 0 else 0.0


def calculate_phrase_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Calculate n-gram phrase overlap."""
    if not text_a or not text_b:
        return 0.0

    def get_ngrams(text, n):
        words = re.findall(r'\w+', text.lower())
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    ngrams_a = set(get_ngrams(text_a, n))
    ngrams_b = set(get_ngrams(text_b, n))

    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)

    return intersection / union if union > 0 else 0.0


def find_matching_phrases(text_a: str, text_b: str, min_length: int = 5) -> list:
    """Find matching phrases between two texts."""
    if not text_a or not text_b:
        return []

    # Get 3-grams and longer
    matches = []

    for n in range(3, 8):  # 3 to 7 word phrases
        def get_ngrams(text, n):
            words = re.findall(r'\w+', text.lower())
            return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

        ngrams_a = set(get_ngrams(text_a, n))
        ngrams_b = set(get_ngrams(text_b, n))

        common = ngrams_a & ngrams_b

        for phrase in common:
            if len(phrase) >= min_length:
                matches.append(phrase)

    # Return unique matches sorted by length
    unique_matches = list(set(matches))
    unique_matches.sort(key=len, reverse=True)

    return unique_matches[:10]  # Top 10 matches
