"""Structural similarity features."""
import re


def calculate_structural_similarity(doc_a: dict, doc_b: dict) -> float:
    """
    Calculate structural similarity between two documents.

    Features:
    - Document length
    - Number of paragraphs
    - Number of sentences
    - Average sentence length
    - Page count
    """
    try:
        text_a = doc_a.get("text", "")
        text_b = doc_b.get("text", "")

        if not text_a or not text_b:
            return 0.0

        # Extract structural features
        features_a = extract_structural_features(doc_a)
        features_b = extract_structural_features(doc_b)

        # Calculate normalized similarity for each feature
        similarities = []

        # Length similarity
        len_a = features_a["length"]
        len_b = features_b["length"]
        if len_a > 0 and len_b > 0:
            len_sim = 1 - abs(len_a - len_b) / max(len_a, len_b)
            similarities.append(len_sim)

        # Paragraph similarity
        para_a = features_a["paragraphs"]
        para_b = features_b["paragraphs"]
        if para_a > 0 and para_b > 0:
            para_sim = 1 - abs(para_a - para_b) / max(para_a, para_b)
            similarities.append(para_sim)

        # Sentence similarity
        sent_a = features_a["sentences"]
        sent_b = features_b["sentences"]
        if sent_a > 0 and sent_b > 0:
            sent_sim = 1 - abs(sent_a - sent_b) / max(sent_a, sent_b)
            similarities.append(sent_sim)

        # Average sentence length similarity
        avg_sent_a = features_a["avg_sentence_length"]
        avg_sent_b = features_b["avg_sentence_length"]
        if avg_sent_a > 0 and avg_sent_b > 0:
            avg_sent_sim = 1 - abs(avg_sent_a - avg_sent_b) / max(avg_sent_a, avg_sent_b)
            similarities.append(avg_sent_sim)

        # Page similarity
        pages_a = doc_a.get("pages", 1)
        pages_b = doc_b.get("pages", 1)
        if pages_a > 0 and pages_b > 0:
            page_sim = 1 - abs(pages_a - pages_b) / max(pages_a, pages_b)
            similarities.append(page_sim)

        # Overall structural similarity
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0

    except:
        return 0.0


def extract_structural_features(doc: dict) -> dict:
    """Extract structural features from a document."""
    text = doc.get("text", "")

    # Character length
    length = len(text)

    # Paragraphs (double newline or single newline with significant spacing)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    num_paragraphs = len(paragraphs)

    # Sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)

    # Average sentence length
    if num_sentences > 0:
        avg_sentence_length = length / num_sentences
    else:
        avg_sentence_length = 0

    return {
        "length": length,
        "paragraphs": num_paragraphs,
        "sentences": num_sentences,
        "avg_sentence_length": avg_sentence_length
    }
