from typing import Union, Optional, Iterable, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import textstat
import spacy
from spacy.matcher import Matcher
import torch
from tqdm import tqdm
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
from nltk import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
from sklearn.feature_selection import mutual_info_classif
from scipy import sparse
import multiprocessing as mp
import time
import hashlib
from functools import partial
from joblib import Parallel, delayed

def process_text_spacy(text: str, nlp: spacy.language.Language) -> str:
    """
    Cleans text using spaCy by removing URLs, special characters, numbers, 
    stopwords, and then applies lemmatization.

    Parameters:
    text (str): The text that needs to be cleaned
    nlp (spacy.Language): The spacy NLP model to handle the cleaning

    Return:
    str: The processed piece of text
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r'\d+|[^a-z\s]', '', text)  # Keep only letters

    doc = nlp(text)  
    words = [token.lemma_ for token in doc 
            if not token.is_stop and token.is_alpha] 
    
    return " ".join(words)

def nltk_clean_text(text: str) -> str:
    """
    Cleans text by removing URLs, special characters, numerics, 
    stopwords, and applying lemmatization while maintaining proper word separation.

    Parameters:
    text (str): The text that needs to be cleaned

    Return:
    str: The processed piece of text
    """
    if not isinstance(text, str):
        return ""

    mention_pattern = r"@[\w\d_]+"  # Matches @username
    hashtag_pattern = r"#[\w\d_]+"  # Matches #hashtag
    media_pattern = r"pic\.twitter\.com/\w+"  # Matches pic.twitter.com URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Matches URLs

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(mention_pattern, "", text)
    text = re.sub(hashtag_pattern, "", text)
    text = re.sub(media_pattern, "", text)
    text = re.sub(url_pattern, "", text)
    
    # First tokenize the text to preserve word boundaries
    words = word_tokenize(text)
    
    # Clean each word individually
    cleaned_words = []
    for word in words:
        # Remove special characters and digits from each word
        clean_word = re.sub(r'\d+|[^a-z]', '', word)
        if clean_word and clean_word not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(clean_word))
    
    return " ".join(cleaned_words)


def nltk_count_word(text: str) -> str:
    """
    Cleans text by removing URLs, special characters, numerics, 
    stopwords, and then count the number of words.

    Parameters:
    text (str): The text that needs to be cleaned

    Return:
    str: The processed piece of text
    """
    if not isinstance(text, str):
        return ""

    mention_pattern = r"@[\w\d_]+"  # Matches @username
    hashtag_pattern = r"#[\w\d_]+"  # Matches #hashtag
    media_pattern = r"pic\.twitter\.com/\w+"  # Matches pic.twitter.com URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Matches URLs

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(mention_pattern, "", text)
    text = re.sub(hashtag_pattern, "", text)
    text = re.sub(media_pattern, "", text)
    text = re.sub(url_pattern, "", text)

    # First tokenize the text to preserve word boundaries
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]

    return len(words)


def extract_ngrams(df: pd.DataFrame,
                        word: str=None,
                        n: int=5,
                        top_k: int=20):
    """
    Extract the most common n-grams containing the word 'word' from a collection of texts.
    
    Parameters:
    df (pd.DataFrame): Pandas dataframe containing all the texts
    word (str): Word the n-gram must contain
    n (int): Size of ngrams (default: 3 for trigrams)
    top_k (int): Number of top ngrams to return
    
    Returns:
    list: Top k most common ngrams containing 'like'
    """

    ngrams_count = {0: Counter(), 1: Counter()}

    for _, row in df.iterrows():
        text = row["text"].lower()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalnum()]

        text_ngrams = list(ngrams(tokens, n))
        if word is not None:
            text_ngrams = [gram for gram in text_ngrams if word in gram]

        ngrams_count[row["label"]].update(text_ngrams)

    total_ngrams = ngrams_count[0] + ngrams_count[1]

    ngrams_df = pd.DataFrame([
    {'ngram': ngram, 'total_count': total_ngrams[ngram], 0: ngrams_count[0][ngram], 1: ngrams_count[1][ngram]}
            for ngram in total_ngrams])

    ngrams_df.sort_values(by="total_count", ascending=False, inplace=True)
    
    if top_k > 0:
        return ngrams_df[:top_k]  
    else:
        return ngrams_df


def extract_empty_short_text(df: pd.DataFrame, 
                             column_name: str = 'text', 
                             min_length: int = 25) -> pd.DataFrame:
    """
    Extracts all rows where the text length in the specified column is below
    the given threshold (`min_length`).

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text column.
    column_name (str): The name of the column containing text.
    min_length (int): The length threshold below which a text is extracted.

    Returns:
    pd.DataFrame: A DataFrame containing only rows where text length is below the threshold.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Ensure column is of string type and handle NaN values
    filtered_df = df[df[column_name].apply(nltk_count_word) <= min_length]

    return filtered_df

def remove_location_line(text: str) -> str:
    """
    Removes the location line from the beginning of a given text.

    The function assumes the location line follows a pattern where it may contain:
    - Words and spaces, possibly followed by a comma and more words. (e.g., "New York (Reuters)") 
    - An optional slash-separated location format. (e.g. "Los Angeles/Santa Fe (Reuters)")
    - An optional parenthetical expression. (e.g. "(Reuters))
    - A sequence of dashes (`-`) acting as a separator. (e.g. Location - (Reuters))

    Parameters:
    text (str): The input text from which the location line should be removed.

    Returns:
    str: The cleaned text with the location line removed.
    """
    return re.sub(r"^(?:[\w\s]+(?:,\s*\w+)?(?:/\s*[\w\s]+(?:,\s*\w+)?)?\s*)?(?:\([^)]*\)\s*)?-+\s*", "", text).strip()


def parse_date(date_str: str) -> Union[np.nan, pd.Timestamp]:
    """
    Parses a date string into a Pandas Timestamp object using multiple formats.

    The function attempts to convert the input string into a date using common date formats:
    - `%d-%b-%y` → e.g., "05-Mar-21"
    - `%B %d, %Y` → e.g., "March 5, 2021"
    - `%B %d, %Y ` (with trailing space) → e.g., "March 5, 2021 "
    - `%b %d, %Y` → e.g., "Mar 5, 2021"
    - `%Y-%m-%d` → e.g., "2021-03-05"
    - `%m/%d/%Y` → e.g., "03/05/2021"

    Parameters:
    date_str (str): The date string to be parsed.

    Returns:
    Union[np.nan, pd.Timestamp]: A Pandas Timestamp object if parsing succeeds, otherwise NaN.
    """
    formats = ['%d-%b-%y', '%B %d, %Y', '%B %d, %Y ', '%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return np.nan

def find_most_common_words(df: pd.DataFrame, 
                           feature: str, 
                           countv: CountVectorizer,
                           date: Optional[pd.Timestamp] = None, 
                           max_features: Optional[int] = None) -> pd.DataFrame:
    """
    Finds the most common words in a given text column of a DataFrame using CountVectorizer.

    The function processes a text column using a CountVectorizer to determine word frequency.
    It allows filtering by date and limiting the number of features.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the text column.
    - feature (str): The column name containing text data (e.g., 'text').
    - countv (CountVectorizer): A scikit-learn CountVectorizer instance.
    - date (Optional[pd.Timestamp]): If provided, filters rows where the "date" column matches 
      this value (e.g., `pd.Timestamp("2024-02-22")`).
    - max_features (Optional[int]): The maximum number of features to extract. If specified, 
      updates `countv.max_features` before processing.

    Returns:
    - pd.DataFrame: A DataFrame with words as the index and their total frequency as values, 
      sorted in descending order.
    """
    if max_features is not None:
        countv.max_features = max_features

    if date is not None:
        df = df[df["date"] == date]
        
    smatrix = countv.fit_transform(df[feature])
    wo_df = (pd.DataFrame(smatrix.toarray(), columns=countv.get_feature_names_out())
               .sum(axis=0)
               .sort_values(ascending=False)
            )
    return wo_df

def extract_spacy_features_batch(texts: Iterable[str], 
                                 nlp: spacy.language.Language, 
                                 batch_size: int=32):

    """
    Extracts a variety of features from a batch of text data using spaCy's NLP processing,
    including token-level features, linguistic features, Twitter-style features, and more.
    
    The function processes a batch of texts to calculate linguistic features such as 
    token counts, part-of-speech ratios, entity recognition ratios, and additional Twitter-style 
    features such as mentions, hashtags, URLs, retweets, and media links.

    Parameters:
    texts (list of str): List of texts to process.
    nlp (spacy.lang): The spaCy language model to use for processing.
    batch_size (int): The number of texts to process in each batch. Default is 32.

    Returns:
    list of dict: A list of dictionaries where each dictionary contains calculated features
                  for a single text. Each dictionary contains ratios and counts for features 
                  like word count, POS ratios, Twitter features, and more.
    """
    
    # Twitter-style elements
    mention_pattern = r"@[\w\d_]+"  # Matches @username
    hashtag_pattern = r"#[\w\d_]+"  # Matches #hashtag
    media_pattern = r"pic\.twitter\.com/\w+"  # Matches pic.twitter.com URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Matches URLs

    features = []
    
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts)):
        total_words = len([token for token in doc if token.is_alpha])
        
        if total_words == 0:
            features.append(create_empty_features())
            continue
            
        # Basic counts
        pos_counts = Counter(token.pos_ for token in doc)
        entity_counts = Counter(ent.label_ for ent in doc.ents)
        stopword_count = sum(1 for token in doc if token.is_stop)
        punct_count = sum(1 for token in doc if token.is_punct)
        
        # Text statistics
        characters_count = len(doc.text)

        # Apply regex patterns to find matches
        mentions = len(re.findall(mention_pattern, doc.text))
        hashtags = len(re.findall(hashtag_pattern, doc.text))
        media = len(re.findall(media_pattern, doc.text))
        urls = len(re.findall(url_pattern, doc.text))

        # Formatting features
        special_chars = len([token for token in doc if token.text.isascii() 
                             and not token.text.isalnum() and not token.is_punct])
        uppercase_words = len([token for token in doc if token.text.isupper() 
                               and len(token.text) > 1])
        line_breaks = doc.text.count('\n')
        
        feature_dict = {
            "words_count": total_words,
            "characters_count": characters_count,
            "person_ratio": entity_counts.get("PERSON", 0) / total_words,
            "gpe_ratio": entity_counts.get("GPE", 0) / total_words,
            "org_ratio": entity_counts.get("ORG", 0) / total_words,
            "date_ratio": entity_counts.get("DATE", 0) / total_words,
            "stopword_ratio": stopword_count / total_words,
            "punct_ratio": punct_count / total_words,
            "avg_token_length": characters_count / total_words if total_words > 0 else 0,
            
            # Twitter features
            "mentions_ratio": mentions / total_words,
            "hashtags_ratio": hashtags / total_words,
            "urls_ratio": urls / total_words,
            "media_ratio": media / total_words,
            
            # Formatting features
            "special_char_num_ratio": special_chars / total_words,
            "uppercase_ratio": len([c for c in doc.text if c.isupper()]) / total_words,
            "uppercase_words_ratio": uppercase_words / total_words,
        }
        
        # Add POS ratios
        pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", 
                    "NUM", "PART", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
        feature_dict.update({
            f"{pos.lower()}_ratio": pos_counts.get(pos, 0) / total_words 
            for pos in pos_tags
        })
        
        features.append(feature_dict)
        
        if len(features) % batch_size == 0:
            torch.cuda.empty_cache()
    
    return features

def get_df_from_sparse(sparse_matrix,
                       original_df: pd.DataFrame,
                       target: str,
                       selected_feature_names: Iterable[float],
                       selected_indices: Iterable[int]=None,
                       ) -> pd.DataFrame:

    """
    Converts a sparse matrix to a pandas DataFrame with selected features and target column.
    
    Parameters:
    sparse_matrix (scipy.sparse matrix): The sparse matrix to convert
    original_df (pd.DataFrame): The original DataFrame used to create the sparse matrix
    selected_indices (Iterable[int]): Indices of the features to select from the sparse matrix
    selected_feature_names (Iterable[str]): Names of the selected features to use as column names
    target (str): Name of the target column in original_df to include in the output DataFrame
    
    Return:
    pd.DataFrame: A DataFrame containing the selected features and the target column
    """
   

    if selected_indices is not None:
        sparse_matrix = sparse_matrix[:, selected_indices]

    feature_df = pd.DataFrame(sparse_matrix.toarray(),
                              columns=selected_feature_names,
                              index=original_df.index
                             )

    feature_df = pd.concat([feature_df, original_df[target]], axis=1)
    
    return feature_df


def parallel_process(df, func, num_workers=32, **func_args):
    """
    A generalized parallel processing function for DataFrames.
    
    Parameters:
    - df: pandas DataFrame
    - func: function to apply to each partition
    - num_workers: number of worker processes
    - func_args: additional arguments to pass to the function
    
    Returns:
    - A concatenated DataFrame with processed results
    """

    df_ = df.copy()
    
    chunks = np.array_split(df_, num_workers)
    
    # Prepare a partial function with function arguments
    func_partial = partial(func, **func_args)
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(func_partial, chunks), total=len(chunks)))

    return pd.concat(results)


def clean_final_text_batch_spacy(df, 
                       nlp,
                       unique_ids=None, 
                       batch_size: int=64,
                       words_to_remove: Iterable[str]=[],
                       multi_nlp: bool=False):
    """
    Process a batch of texts with spaCy, performing entity anonymization
    and text cleaning in a single pass.
    
    This version ensures entity replacement happens correctly.
    """
    
    processed_results = []
    
    # Handle None/NaN values before sending to spaCy
    valid_texts = []
    valid_indices = []

    if isinstance(df, pd.DataFrame):
        texts = df["text"]
    elif isinstance(df, pd.Series):
        texts = df

    if unique_ids is None:
        unique_ids = [None] * len(texts)
    
    for i, text in enumerate(texts):
        if isinstance(text, str):
            valid_texts.append(text)
            valid_indices.append(i)
        else:
            # Add placeholder for invalid texts to maintain order
            processed_results.append("")
    
    # Process valid texts with spaCy and track the mapping to original indices
    if multi_nlp:
        nlp_ = spacy.load(nlp.meta["lang"] + "_" + nlp.meta["name"])
        docs = list(nlp_.pipe(valid_texts, batch_size=batch_size))
    else:
        docs = list(nlp.pipe(valid_texts, batch_size=batch_size))
    
    # Process each valid document
    for doc_idx, doc in enumerate(docs):
        orig_idx = valid_indices[doc_idx]
        text = valid_texts[doc_idx]
        unique_id = unique_ids[orig_idx]
        
        # First handle entity anonymization
        # Track entity replacements for this text
        entity_map = {}
        # Make a copy of the text for replacement
        anonymized_text = text
        
        # Extract all entities with their positions
        entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) 
                   for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
        
        # Sort entities in reverse order to avoid position shifts
        entities.sort(key=lambda x: x[1], reverse=True)
        
        # Replace entities in the text
        for entity_text, start, end, label in entities:
            # If we've seen this entity before, use the same replacement
            if entity_text in entity_map:
                replacement = entity_map[entity_text]
            else:
                # Create a hash of the entity
                if unique_id is not None:
                    hash_input = f"{entity_text.lower()}_{unique_id}"
                else:
                    hash_input = f"{entity_text.lower()}_{orig_idx}"
                
                hash_obj = hashlib.md5(hash_input.encode())
                hash_id = hash_obj.hexdigest()[:6]  # short hash
                replacement = f"{label}{hash_id}"
                entity_map[entity_text] = replacement
            
            # Replace the entity in the text
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
        
        # Now apply the cleaning steps to the anonymized text
        # Create a new doc for the anonymized text to get proper lemmatization
        # This is more reliable than trying to use the original doc after replacements
        clean_doc = nlp(anonymized_text.lower())
        
        # Apply regex cleaning
        mention_pattern = r"@[\w\d_]+"
        hashtag_pattern = r"#[\w\d_]+"
        media_pattern = r"pic\.twitter\.com/\w+"
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        possession_pattern = r"\b(\w+)'s\b"
        
        cleaned_text = anonymized_text.lower()
        cleaned_text = re.sub(mention_pattern, "", cleaned_text)
        cleaned_text = re.sub(hashtag_pattern, "", cleaned_text)
        cleaned_text = re.sub(media_pattern, "", cleaned_text)
        cleaned_text = re.sub(url_pattern, "", cleaned_text)
        cleaned_text = re.sub(possession_pattern, r"\1", cleaned_text)
        cleaned_text = re.sub(r"\b\w{1,2}\b", "", cleaned_text)
        
        # Extract lemmatized tokens from the cleaned doc
        tokens = [token.lemma_.lower() for token in clean_doc 
                 if token.text.isalnum() and not token.is_stop and token.lemma_.lower() not in words_to_remove]
        
        processed_results.append(" ".join(tokens))

    if isinstance(df, pd.DataFrame):
        df["text"] = processed_results
    else:
        df = processed_results
    
    return df

def extract_linguistic_features(df: pd.DataFrame):
    """
    Extracts various linguistic readability features from the provided text in the DataFrame.
    
    The function applies several readability formulas and measures to the text column 
    and adds the results as new columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing a column 'text' which holds the text data 
                            to analyze.

    Returns:
    pandas.DataFrame: The original DataFrame with additional columns for each linguistic feature. 
                       The new columns include readability scores such as Flesch Reading Ease, 
                       Flesch-Kincaid Grade, SMOG Index, Coleman-Liau Index, and others.
    """
    # Applying readability formulas to the 'text' column and storing results in new columns
    df["flesch_reading_ease"] = df["text"].apply(textstat.flesch_reading_ease)
    df["flesch_kincaid_grade"] = df["text"].apply(textstat.flesch_kincaid_grade)
    df["smog_index"] = df["text"].apply(textstat.smog_index)
    df["coleman_liau_index"] = df["text"].apply(textstat.coleman_liau_index)
    df["automated_readability_index"] = df["text"].apply(textstat.automated_readability_index)
    df["dale_chall_readability_score"] = df["text"].apply(textstat.dale_chall_readability_score)
    df["difficult_words"] = df["text"].apply(textstat.difficult_words)
    df["gunning_fog"] = df["text"].apply(textstat.gunning_fog)

    return df


def extract_spacy_features_batch(df: pd.DataFrame,
                                 nlp,
                                 batch_size: int=32):

    """
    Extracts a variety of features from a batch of text data using spaCy's NLP processing,
    including token-level features, linguistic features, Twitter-style features, and more.
    
    The function processes a batch of texts to calculate linguistic features such as 
    token counts, part-of-speech ratios, entity recognition ratios, and additional Twitter-style 
    features such as mentions, hashtags, URLs, retweets, and media links.

    Parameters:
    texts (list of str): List of texts to process.
    nlp (spacy.lang): The spaCy language model to use for processing.
    batch_size (int): The number of texts to process in each batch. Default is 32.

    Returns:
    list of dict: A list of dictionaries where each dictionary contains calculated features
                  for a single text. Each dictionary contains ratios and counts for features 
                  like word count, POS ratios, Twitter features, and more.
    """

    texts = df["text"]
    
    # Twitter-style elements
    mention_pattern = r"@[\w\d_]+"  # Matches @username
    hashtag_pattern = r"#[\w\d_]+"  # Matches #hashtag
    media_pattern = r"pic\.twitter\.com/\w+"  # Matches pic.twitter.com URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Matches URLs

    features = []

    for doc in list(nlp.pipe(texts, batch_size=batch_size)):
        total_words = len([token for token in doc if token.is_alpha])

        if total_words == 0:
            features.append(create_empty_features())
            continue

        # Basic counts
        pos_counts = Counter(token.pos_ for token in doc)
        entity_counts = Counter(ent.label_ for ent in doc.ents)
        stopword_count = sum(1 for token in doc if token.is_stop)
        punct_count = sum(1 for token in doc if token.is_punct)

        # Text statistics
        characters_count = len(doc.text)

        # Apply regex patterns to find matches
        mentions = len(re.findall(mention_pattern, doc.text))
        hashtags = len(re.findall(hashtag_pattern, doc.text))
        media = len(re.findall(media_pattern, doc.text))
        urls = len(re.findall(url_pattern, doc.text))

        # Formatting features
        special_chars = len([token for token in doc if token.text.isascii()
                             and not token.text.isalnum() and not token.is_punct])
        uppercase_words = len([token for token in doc if token.text.isupper()
                               and len(token.text) > 1])
        line_breaks = doc.text.count('\n')

        feature_dict = {
            "words_count": total_words,
            "characters_count": characters_count,
            "person_ratio": entity_counts.get("PERSON", 0) / total_words,
            "gpe_ratio": entity_counts.get("GPE", 0) / total_words,
            "org_ratio": entity_counts.get("ORG", 0) / total_words,
            "date_ratio": entity_counts.get("DATE", 0) / total_words,
            "stopword_ratio": stopword_count / total_words,
            "punct_ratio": punct_count / total_words,
            "avg_token_length": characters_count / total_words if total_words > 0 else 0,

            # Twitter features
            "mentions_ratio": mentions / total_words,
            "hashtags_ratio": hashtags / total_words,
            "urls_ratio": urls / total_words,
            "media_ratio": media / total_words,

            # Formatting features
            "special_char_num_ratio": special_chars / total_words,
            "uppercase_ratio": len([c for c in doc.text if c.isupper()]) / total_words,
            "uppercase_words_ratio": uppercase_words / total_words,
        }

        # Add POS ratios
        pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                    "NUM", "PART", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
        feature_dict.update({
            f"{pos.lower()}_ratio": pos_counts.get(pos, 0) / total_words
            for pos in pos_tags
        })

        features.append(feature_dict)

    features_df = pd.DataFrame(features, index=df.index)

    return features_df

def create_empty_features():
    """Create a dictionary of empty features for error cases"""

    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                "NUM", "PART", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]

    return {
        "words_count": 0,
        "characters_count": 0,
        "person_ratio": 0,
        "gpe_ratio": 0,
        "org_ratio": 0,
        "date_ratio": 0,
        "stopword_ratio": 0,
        "punct_ratio": 0,
        "avg_token_length": 0,
        "mentions_ratio": 0,
        "hashtags_ratio": 0,
        "urls_ratio": 0,
        "retweets_ratio": 0,
        "special_char_num_ratio": 0,
        "uppercase_ratio": 0,
        "uppercase_words_ratio": 0,
        "line_breaks": 0,
        **{f"{pos.lower()}_ratio": 0 for pos in pos_tags}
    }

def extract_all_features(df: pd.DataFrame, 
                         nlp,
                         batch_size: int=32):
    """
    Main function to extract all features using the provided spaCy model.
    
    This function extracts both spaCy-specific linguistic features and additional 
    custom features related to Twitter-style elements (e.g., mentions, hashtags, 
    retweets) and formatting (e.g., uppercase words, special characters).
    
    The function processes the given dataframe with a column of text, applies 
    the spaCy language model to extract linguistic features, and combines them 
    with additional feature sets related to readability, Twitter features, and formatting.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the text to process.
    batch_size (int): The number of texts to process in each batch. Default is 32.

    Returns:
    pandas.DataFrame: A dataframe containing the extracted features, with the original 
                       dataframe index preserved.
    """
    

    df_ = df.copy()
    spacy_features_df = extract_spacy_features_batch(df_, nlp, batch_size)

    # Extract linguistic features
    linguistic_features = extract_linguistic_features(df_)

    all_features_df = pd.concat([linguistic_features, spacy_features_df], axis=1)    
    return all_features_df

def extract_alternative_features(df: pd.DataFrame):

    df_ = df.copy()
    sia = SentimentIntensityAnalyzer()
    df_["polarity"] = df_["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_["subjectivity"] = df_["text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df_["vader_compound"] = df_["text"].apply(lambda x: sia.polarity_scores(x)["compound"])

    df_["nrc_emotions"] = df_["text"].apply(lambda x: NRCLex(x).raw_emotion_scores)

    # Extract Specific Emotion Scores
    df_["anger"] = df_["nrc_emotions"].apply(lambda x: x.get("anger", 0))
    df_["joy"] = df_["nrc_emotions"].apply(lambda x: x.get("joy", 0))
    df_["sadness"] = df_["nrc_emotions"].apply(lambda x: x.get("sadness", 0))
    df_["fear"] = df_["nrc_emotions"].apply(lambda x: x.get("fear", 0))

    df_["anticipation"] = df_["nrc_emotions"].apply(lambda x: x.get("anticipation", 0))
    df_["trust"] = df_["nrc_emotions"].apply(lambda x: x.get("trust", 0))
    df_["surprise"] = df_["nrc_emotions"].apply(lambda x: x.get("surprise", 0))
    df_["disgust"] = df_["nrc_emotions"].apply(lambda x: x.get("disgust", 0))

    # Drop the full dictionary column if not needed
    df_ = df_.drop(columns=["nrc_emotions"])

    df_ = extract_linguistic_features(df_)

    df_.drop(["text"], axis=1, inplace=True)

    return df_
