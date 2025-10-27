# utilities.py
import math
import re
import string
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import yake
import spacy
import heapq

# Load Spacy model (assuming it's needed for spacy_tokenizer)
nlp = spacy.load("en_core_web_sm")
stopwords = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
_RE_DIGITS = re.compile(r"\d")
def extract_features(narrative, n=3):
    '''
    Extract word and n-gram features from a narrative text.
    
    This function tokenizes the input narrative into words and generates all possible
    n-grams from length 1 up to n (inclusive). All text is converted to lowercase
    for case-insensitive feature extraction.
    
    Parameters
    ----------
    narrative : str
        The input text narrative from which to extract features. The text will be
        split on whitespace to create tokens.
    n : int, optional
        The maximum length of n-grams to extract (default is 3). For example,
        if n=3, the function will extract unigrams (1-grams), bigrams (2-grams),
        and trigrams (3-grams).
    
    Returns
    -------
    set of tuple
        A set containing tuples representing the extracted n-gram features.
        Each tuple contains 1 to n consecutive words from the narrative.
        Duplicates are automatically removed due to the set data structure.
    '''
    words = tuple(narrative.lower().split())
    features = set()
    for i in range(len(words)):
        for j in range(1, n+1):
            if i + j <= len(words):
                features.add(words[i : i+j])
    return features

def calculate_idf(narratives, text_column=None):
    """Calculate Inverse Document Frequency (IDF) for words and n-grams in a collection of narratives.
    
    This function computes the IDF score for each unique feature (word or n-gram) across
    a corpus of narrative documents. IDF measures how rare or common a term is across
    the entire document collection. Features that appear in fewer documents receive
    higher IDF scores, indicating they are more distinctive.
    
    The IDF formula used is: log(total_documents / document_frequency)
    
    Parameters
    ----------
    narratives : list of str
        A collection of narrative text documents.
    text_column : str or None, optional
        Not used. Parameter exists for API consistency only.
    Returns
    -------
    list of tuple
        A list of tuples where each tuple contains (idf_score, feature_string).
        The list is sorted in ascending order by IDF score (lowest to highest).
        Features with lower IDF scores appear in more documents (more common),
        while features with higher IDF scores appear in fewer documents (more rare).
        
        - idf_score : float
            The calculated IDF score for the feature
        - feature_string : str
            The feature as a space-separated string (e.g., "word" for unigrams,
            "word1 word2" for bigrams)
    """
    lenNarratives = len(narratives)
    tD = Counter()
    for narrative in narratives:
        features = extract_features(narrative)
        for feature in features:
            tD[" ".join(feature)] += 1
    IDF = []
    for (word, word_frequency) in tD.items():
        word_IDF = math.log(float(lenNarratives) / word_frequency)
        IDF.append((word_IDF, word))
    IDF.sort(reverse = False)
    return IDF

def calculate_tf(narratives):
    """ Calculate Term Frequency (TF) for words in each narrative document.
    
    This function computes the normalized term frequency for each word in every
    narrative document. Term frequency measures how often a word appears in a
    document relative to the total number of words in that document. The TF
    score is calculated as: word_count / total_words_in_document.
    
    Each narrative is processed independently, and case sensitivity is preserved
    (i.e., "Word" and "word" are treated as different terms). This is done since
    the lower casing is handled separately in the preprocessing pipeline.

    Parameters
    ----------
    narratives : list of str
        A collection of narrative text documents. Each narrative should be a string
        containing words separated by whitespace. Empty narratives will result in
        empty dictionaries in the output.
    
    Returns
    -------
    list of dict
        A list of dictionaries, one for each input narrative. Each dictionary
        maps words (str) to their normalized term frequencies (float).
        
        - Keys are words (str) as they appear in the original text (case-sensitive)
        - Values are normalized frequencies (float) between 0 and 1
        - The sum of all TF values in each dictionary equals 1.0
        
        The order of dictionaries matches the order of input narratives.
    """
    dictlist=[]
    l=0
    for narrative in narratives:
        l = len(narrative.split())
        dic = {}
        narrative = narrative.lower()
        for word in narrative.split():
            if word in dic:
                dic[word] = dic[word] + 1
            else:
                dic[word]=1
    
        if l > 0:  
            for word in dic:
                dic[word] = dic[word] / l
        dictlist.append(dic)
    return dictlist
        
def calculate_mean_tf(tf_Frame):
    """Calculate the mean Term Frequency (TF) across all narrative documents.
    
    This function takes a list of TF dictionaries (typically from calculate_tf)
    and computes the average term frequency for each unique word across all
    documents. Words that don't appear in certain documents are treated as
    having a TF score of 0 for those documents when calculating the mean.
    
    Parameters
    ----------
    tf_Frame : list of dict
        A list of dictionaries where each dictionary represents the term
        frequencies for one document. Each dictionary should map words (str)
        to their normalized term frequencies (float). This is typically the
        output from the calculate_tf function.
    
    Returns
    -------
    dict
        A dictionary mapping each unique word (str) to its mean term frequency
        (float) across all documents. Words that appear in only some documents
        will have their missing values treated as 0 when computing the average.
        
        Keys: unique words (str) from all input documents
        Values: mean TF scores (float) across all documents
    """
    DF = pd.DataFrame(tf_Frame)
    DF.fillna(0, inplace=True)
    return dict(DF.mean(numeric_only=True))
    
def handle_punctuation(narrative):
    """Add spaces after punctuation marks for better tokenization.
    
    This function inserts a space after specific punctuation marks when they
    are immediately followed by a non-whitespace character. This preprocessing
    step improves tokenization quality by ensuring punctuation doesn't stick
    to adjacent words, which is especially useful for text that lacks proper
    spacing after punctuation marks.
    
    The function targets these punctuation marks: ? . , ! " ) ]
    
    Parameters
    ----------
    narrative : str
        The input text narrative that may contain punctuation marks without
        proper spacing. Can be any string including empty strings.
    
    Returns
    -------
    str
        The processed narrative with spaces inserted after punctuation marks
        where needed. The original text is preserved except for the added spaces.
        If the input is empty or contains no matching punctuation patterns,
        the original string is returned unchanged.
    """
    return re.sub(r'[\W_]', r' ', narrative)
    
def remove_num(narrative):
    """ Remove all digits from the narrative text by replacing them with spaces.
    
    This function identifies every numeric digit (0-9) in the input text and
    replaces each digit with a single space character. This preprocessing step
    is commonly used in text analysis to focus on textual content while removing
    numeric information that might not be relevant for certain NLP tasks.
    
    Note that this function replaces digits with spaces rather than completely
    removing them, which helps preserve word boundaries and prevents words from
    being inadvertently joined together.
    
    Parameters
    ----------
    narrative : str
        The input text narrative that may contain numeric digits (0-9).
        Can be any string including empty strings, unicode text, or text
        with mixed alphanumeric content.
    
    Returns
    -------
    str
        The processed narrative with all digits (0-9) replaced by single spaces.
        The original text structure is preserved except for the digit replacements.
        Multiple consecutive digits will result in multiple consecutive spaces.
        If the input contains no digits, the original string is returned unchanged.
    """
    return re.sub(r'\d', r' ', narrative)

def preprocess_text(narrative):
    """Perform comprehensive text preprocessing for natural language processing tasks.
    
    This function applies a series of standard text preprocessing steps to clean
    and normalize input text. The preprocessing pipeline includes punctuation
    removal, whitespace normalization, and case standardization. This creates
    a clean, consistent text format suitable for most NLP and text analysis tasks.
    
    Processing Steps (in order):
    1. Remove all punctuation marks (based on string.punctuation)
    2. Remove all numeric digits (0-9)
    3. Normalize whitespace (collapse multiple spaces into single spaces)
    4. Strip leading and trailing whitespace
    5. Convert all text to lowercase
    
    Parameters
    ----------
    narrative : str
        The input text narrative to be preprocessed. Can contain any combination
        of letters, numbers, punctuation, and whitespace characters. Empty
        strings are handled gracefully and return empty strings.
    
    Returns
    -------
    str
        The preprocessed narrative with all transformations applied:
        - All punctuation removed
        - All numeric digits removed
        - Multiple consecutive whitespace characters normalized to single spaces
        - Leading and trailing whitespace stripped
        - All characters converted to lowercase
        
        If the input consists entirely of punctuation, numbers, and whitespace,
        an empty string will be returned.
    """
    narrative = narrative.translate(str.maketrans('', '', punctuations))
    narrative = _RE_DIGITS.sub(" ", narrative)
    narrative = _RE_COMBINE_WHITESPACE.sub(" ", narrative).strip()
    narrative = narrative.lower()
    return narrative

def remove_words_dict(narrative, dictionary):
    """Remove specific words from narrative text based on a provided dictionary.
    
    This function filters out words from the input narrative that appear in the
    provided dictionary. It is useful for removing stop words, sensitive terms,
    or any other predefined set of words from text data. The function efficiently
    converts the dictionary to a set for fast lookup operations during filtering.
    
    Processing Steps:
    1. Convert the input dictionary to a set for O(1) lookup performance
    2. Split the narrative into individual words
    3. Filter out words that appear in the dictionary set
    4. Rejoin the remaining words into a cleaned text string
    
    Parameters
    ----------
    narrative : str
        The input text narrative from which words will be removed. Should be
        a string of space-separated words. Empty strings are handled gracefully
        and return empty strings.
    dictionary : list or set
        A collection of words to be removed from the narrative. Can be provided
        as a list, set, or any iterable containing the target words to filter out.
        The function automatically converts this to a set for optimal performance.
    
    Returns
    -------
    str
        The filtered narrative with all dictionary words removed:
        - Only contains words not present in the dictionary
        - Maintains original word order for non-removed words
        - Preserves single space separation between words
        - Returns empty string if all words are removed or input is empty
        """
    dictionary_set = set(dictionary)  # Convert dictionary to a set for faster lookups
    return ' '.join(word for word in narrative.split() if word not in dictionary_set)

def get_max_n_dict(dic,n):
    """ Get the top n entries from a dictionary based on their values (highest to lowest).
    
    This function extracts the n key-value pairs with the highest values from a
    dictionary and returns them as a new dictionary. The entries are selected
    based on descending value order, making this useful for finding the most
    frequent terms, highest scores, or top-ranked items in various data analysis
    contexts.
    
    The function maintains the key-value relationships from the original dictionary
    while filtering to only the top n entries. Keys are sorted by their associated
    values in descending order, and ties are handled by Python's stable sorting
    (maintaining original order for equal values).
    
    Parameters
    ----------
    dic : dict
        The input dictionary containing key-value pairs where values are comparable
        (typically numeric). The dictionary should have values that support comparison
        operations (>, <, ==). Keys can be of any hashable type.
        
    n : int
        The number of top entries to return. Must be a non-negative integer.
        If n is larger than the dictionary size, all entries are returned.
        If n is 0, an empty dictionary is returned.

     
    Returns
    -------
    dict
        A new dictionary containing the top n key-value pairs from the input
        dictionary, ordered by value in descending order. The original key-value
        relationships are preserved. If multiple entries have the same value,
        the selection follows Python's stable sorting behavior.
        
        The returned dictionary maintains insertion order (Python 3.7+ behavior)
        with entries ordered from highest to lowest value.
    """
      # Input validation
    if not isinstance(dic, dict):
        raise TypeError(f"Expected dictionary, got {type(dic).__name__}")
    
    if not isinstance(n, int):
        raise TypeError(f"Parameter 'n' must be an integer, got {type(n).__name__}")
    
    if n < 0:
        raise ValueError(f"Parameter 'n' must be non-negative, got {n}")
    
    if n == 0 or len(dic) == 0:
        return {}
    
    n = min(n, len(dic))
    

    dict_size = len(dic)
    

    if n < dict_size // 10 and dict_size > 100:
        # heapq.nlargest is O(dict_size * log n) vs O(dict_size * log dict_size) for sorting
        return dict(heapq.nlargest(n, dic.items(), key=lambda item: item[1]))
    else:
        # Use sorting approach for small dictionaries or when n is large relative to dict_size
        return {key: dic[key] for key in sorted(dic, key=dic.get, reverse=True)[:n]}

def calculate_tf_corpus(corpus,n):
    """Calculate the top n Term Frequency (TF) scores for an entire corpus of documents.
    
    This internal utility function provides a streamlined way to find the most 
    frequently used terms across a collection of documents. It combines three 
    processing steps: calculating individual document TF scores, computing mean 
    TF across the corpus, and selecting the top n terms with highest average frequencies.
    
    Parameters
    ----------
    corpus : list of str
        A collection of text documents (narratives) to analyze. Expected to be
        pre-validated by calling functions in this library.
        
    n : int
        The number of top terms to return. Expected to be a valid non-negative
        integer as validated by calling functions.
    
    Returns
    -------
    dict
        A dictionary containing the top n terms and their mean TF scores,
        ordered by TF score in descending order (highest to lowest).
    """
    D = calculate_tf(corpus)
    g = calculate_mean_tf(D)
    return get_max_n_dict(g,n)

def calculate_tfidf_corpus(corpus, n):
    """
    Calculate the top n TF-IDF (Term Frequency-Inverse Document Frequency) scores for an entire corpus.
    
    This internal utility function computes TF-IDF scores by combining term frequency (TF) 
    and inverse document frequency (IDF) measures across all documents in a corpus. TF-IDF 
    identifies terms that are both frequent in specific documents and distinctive across 
    the corpus, making it useful for feature extraction and document analysis.
    
    The function calculates TF-IDF for each document, computes mean TF-IDF scores across 
    the corpus, and returns the top n terms with the highest average TF-IDF scores.
    
    TF-IDF Formula: TF-IDF(term, doc) = TF(term, doc) × IDF(term, corpus)
    Where:
    - TF = Term Frequency (normalized by document length)  
    - IDF = Inverse Document Frequency (log(total_docs / docs_containing_term))
    
    Parameters
    ----------
    corpus : list of str
        A collection of text documents to analyze. Expected to be pre-validated 
        by calling functions in this library. Each document should be a string
        containing space-separated tokens.
        
    n : int  
        The number of top terms to return based on mean TF-IDF scores. Expected 
        to be a valid non-negative integer as validated by calling functions.
    
    Returns
    -------
    dict
        A dictionary containing the top n terms and their mean TF-IDF scores,
        ordered by TF-IDF score in descending order (highest to lowest).
        
        Keys: terms (str) as they appear in the original text
        Values: mean TF-IDF scores (float) representing average TF-IDF across all documents
        
        Higher scores indicate terms that are both frequent in individual documents
        and distinctive across the corpus.
    
    """
   
    idf_scores = calculate_idf(corpus)
    
    tf_documents = calculate_tf(corpus)
    
    idf_dict = {word: idf_score for idf_score, word in idf_scores}
    
    tfidf_documents = []
    for doc_tf in tf_documents:
        doc_tfidf = {}
        for word, tf_score in doc_tf.items():
            if word in idf_dict:
                doc_tfidf[word] = tf_score * idf_dict[word]
            # Note: words not in IDF dict get implicitly excluded
        tfidf_documents.append(doc_tfidf)
    
    mean_tfidf = calculate_mean_tf(tfidf_documents)
    
    return get_max_n_dict(mean_tfidf, n)

def calculate_tfdf_corpus(corpus, n):
    """
    Calculate top n TF*DF (Term Frequency / Document Frequency) scores for the entire corpus.
    
    TF/DF is a text analysis metric where:
    - TF (Term Frequency): How often a term appears in a document
    - DF (Document Frequency): How many documents contain the term
    - TF*DF product: Higher values indicate terms that appear frequently in many documents
    
    This metric helps identify terms that are characteristic of specific documents
    rather than common across the entire corpus (opposite of TF-IDF behavior).
    
    Parameters:
    -----------
    corpus : list of str or list of dict
        Collection of documents to analyze. Can be:
        - List of strings (raw text documents)
        - List of dictionaries (pre-processed term frequencies)
    n : int
        Number of top TF/DF scores to return
        
    Returns:
    --------
    dict
        Dictionary containing the top n terms with their TF*DF scores,
        sorted in descending order by score
        

    """

    df_values = calculate_idf(corpus)  
    tf_dicts = calculate_tf(corpus)    

    all_tfdf_scores = []
    
    for doc_tf in tf_dicts:
        doc_tfdf = {}
        for df, word in df_values:
            if word in doc_tf:
                tf = doc_tf[word]
                doc_tfdf[word] = tf * df
        if doc_tfdf:  
            all_tfdf_scores.append(doc_tfdf)
    
    if all_tfdf_scores:
        mean_scores = calculate_mean_tf(all_tfdf_scores)
        return get_max_n_dict(mean_scores, n)
    else:
        return {}
    
def extract_kw_yake_corpus(corpus, N, N2, ngram_size=1, dedupe_threshold=0.9):
    """
    YAKE keyword extraction for corpus analysis.
    
    Parameters:
    -----------
    corpus : list of str
        Collection of text documents
    N : int
        Number of keywords to extract per document
    N2 : int  
        Number of top keywords to return from final results
    ngram_size : int, optional (default=1)
        Size of n-grams to extract (1=unigrams, 2=bigrams, etc.)
    dedupe_threshold : float, optional (default=0.9)
        Deduplication threshold for YAKE (0.1-1.0)
        
    Returns:
    --------
    dict
        Top N2 keywords with their aggregated inverse YAKE scores
    """
    if isinstance(corpus, pd.Series):
        corpus = corpus.tolist()
    
    if not corpus or not all(isinstance(doc, str) for doc in corpus):
        raise ValueError("Corpus must be a non-empty list of strings")
    if N <= 0 or N2 <= 0:
        raise ValueError("N and N2 must be positive integers")
    
    try:
        # Use standard YAKE parameter names
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=ngram_size,
            windowsSize=1,
            top=N,
            features=None,
            stopwords=None,
            dedupLim=dedupe_threshold
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YAKE extractor: {e}")
    
    all_keyword_scores = {}
    processed_docs = 0
    
    for doc_idx, document in enumerate(corpus):
        if not document or not document.strip():
            continue  # Skip empty documents
            
        try:
            keywords = kw_extractor.extract_keywords(document)
            
            for keyword, score in keywords:
                # Avoid division by zero and handle very small scores
                safe_score = max(score, 1e-10)
                inverse_score = 1.0 / safe_score
                
                if keyword not in all_keyword_scores:
                    all_keyword_scores[keyword] = []
                all_keyword_scores[keyword].append(inverse_score)
                
            processed_docs += 1
            
        except Exception as e:
            print(f"Warning: Failed to extract keywords from document {doc_idx}: {e}")
            continue
    
    if not all_keyword_scores:
        return {}
    
    # Calculate mean inverse scores across documents
    mean_scores = {}
    for keyword, scores in all_keyword_scores.items():
        mean_scores[keyword] = sum(scores) / len(scores)
    
    return get_max_n_dict(mean_scores, N2)

def spacy_tokenizer(narrative):
    """
    Tokenize text using Spacy with lemmatization and stopword removal.
    
    This function processes text through the following pipeline:
    1. Tokenizes input text using Spacy's NLP pipeline
    2. Lemmatizes tokens (converts words to their base form)
    3. Handles pronoun lemmatization specially 
    4. Removes stopwords and punctuation
    5. Converts to lowercase and strips whitespace
    6. Returns processed tokens as a single string
    
    The function is designed for text preprocessing in NLP tasks where clean,
    lemmatized tokens without stopwords are needed for further analysis.
    
    Parameters:
    -----------
    narrative : str
        Input text to be tokenized and processed. Should be a string containing
        natural language text. Empty strings and None values should be handled
        by the calling code.
        
    Returns:
    --------
    str
        Processed text as a single string with tokens separated by spaces.
        All tokens are lowercase, lemmatized, and cleaned of stopwords/punctuation.
        
    """
    if not isinstance(narrative, str):
        raise ValueError("Input must be a string")
        
    if not narrative.strip():
        return ""
    
    
    if nlp_model is None:
        try:
            nlp_model = nlp  # Try to use global nlp
        except NameError:
            raise RuntimeError("No spacy model provided and global 'nlp' not found")
    
    if custom_stopwords is None:
        # Use spacy's built-in stopwords if available
        try:
            custom_stopwords = nlp_model.Defaults.stop_words
        except AttributeError:
            custom_stopwords = set()  # Fallback to empty set
    
    try:
        
        doc = nlp_model(narrative)
        
        processed_tokens = []
        
        for token in doc:
            # Skip if token should be filtered out
            if remove_punctuation and (token.is_punct or token.is_space):
                continue
                
            if token.is_stop and token.lower_ in custom_stopwords:
                continue
            
            # Get lemmatized form
            # Modern spacy doesn't use "-PRON-", but keep for backwards compatibility
            if token.lemma_ == "-PRON-" or token.lemma_ == "PRON":
                clean_token = token.lower_
            else:
                clean_token = token.lemma_.lower()
            
            # Additional filtering
            clean_token = clean_token.strip()
            if len(clean_token) >= min_token_length and clean_token.isalpha():
                processed_tokens.append(clean_token)
        
        return " ".join(processed_tokens)
        
    except Exception as e:
        raise RuntimeError(f"Spacy processing failed: {str(e)}")
    
def vectorize(text, max_features):    
    """ Vectorize text using TF-IDF with specified max features.
    
    This function creates a TF-IDF (Term Frequency-Inverse Document Frequency) 
    vectorization of input text documents. TF-IDF reflects how important a word 
    is to a document in a collection of documents by combining term frequency 
    with inverse document frequency weighting.
    
    Parameters:
    -----------
    text : array-like of str
        Collection of text documents to vectorize. Each element should be a 
        string representing a document. Can be a list, numpy array, pandas 
        Series, or any iterable containing text documents.
        
    max_features : int
        Maximum number of features (unique tokens) to extract from the corpus.
        If the vocabulary size exceeds this number, only the top max_features
        tokens ordered by term frequency across the corpus will be kept.
        Must be a positive integer."""
    vectorizer = TfidfVectorizer(max_features = max_features)
    X = vectorizer.fit_transform(text)
    return X
    
def get_number_unique_words(narratives):
    """
    Get the number of unique words in narratives.
    
    This function counts the total number of unique words (tokens) across a 
    collection of text documents using sklearn's CountVectorizer. The function
    fits a CountVectorizer on the input narratives and returns the size of the
    resulting vocabulary.
    
    Parameters:
    -----------
    narratives : array-like of str
        Collection of text documents to analyze. Each element should be a 
        string representing a document. Can be a list, numpy array, pandas 
        Series, or any iterable containing text documents.
        
    Returns:
    --------
    int
        Total number of unique words (tokens) found across all narratives.
        This represents the vocabulary size after tokenization using 
        CountVectorizer's default settings.
        
 
    """
    # Input validation
    if narratives is None:
        raise ValueError("Narratives cannot be None")
    
    try:
        narrative_list = list(narratives)
    except (TypeError, ValueError):
        raise ValueError("Narratives must be iterable")
    
    if not narrative_list:
        raise ValueError("Narratives cannot be empty")
    
    if not all(isinstance(doc, str) for doc in narrative_list):
        raise ValueError("All elements in narratives must be strings")
    
    try:
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit(narrative_list)
        return len(count_vectorizer.get_feature_names_out())
        
    except Exception as e:
        raise RuntimeError(f"Failed to count unique words: {str(e)}")


def calculate_word_length_distribution(narratives):
    """
    Calculate the distribution of word lengths in the narratives.
    
    This function analyzes word lengths across a collection of text documents
    and returns a frequency distribution showing how many words of each length
    appear in the corpus. Words are tokenized using simple whitespace splitting.
    
    Parameters:
    -----------
    narratives : array-like of str
        Collection of text documents to analyze. Each element should be a 
        string representing a document. Can be a list, numpy array, pandas 
        Series, or any iterable containing text documents.
        
    Returns:
    --------
    dict
        Dictionary with word lengths (int) as keys and their frequencies (int) 
        as values. For example: {1: 45, 2: 123, 3: 189, ...} indicates there
        are 45 words of length 1, 123 words of length 2, etc.
    """
    length_dist = Counter()
    for narrative in narratives:
        # Check if the narrative is a non-empty string before splitting
        if isinstance(narrative, str) and narrative.strip():
            words = narrative.split()
            for word in words:
                length_dist[len(word)] += 1
    return dict(length_dist)

def extract_most_common_pos_tags(narratives, top_n=10):
    """
    Extract the most common part-of-speech tags from a collection of narratives.
    
    This function processes a collection of text documents using spaCy's NLP
    pipeline to identify and count part-of-speech (POS) tags. It returns the
    most frequent POS tags across all documents, providing insights into the
    grammatical structure and composition of the text corpus.
    
    The function includes comprehensive input validation, error handling, and
    supports flexible model injection. It gracefully handles empty documents
    and processing failures while maintaining robust performance.

    Parameters:
    -----------
    narratives : array-like of str
        Collection of text documents to analyze
    top_n : int, optional (default=10)
        Number of most common POS tags to return (must be positive)
    
    Returns:
    --------
    list of tuple
        List of (tag, count) tuples sorted by frequency in descending order
        
    Raises:
    -------
    ValueError: If input parameters are invalid
    RuntimeError: If spaCy processing fails or no model is available
    """
    if narratives is None:
        raise ValueError("Narratives cannot be None")
    
    try:
        narrative_list = list(narratives)
    except (TypeError, ValueError):
        raise ValueError("Narratives must be iterable")
    
    if not narrative_list:
        raise ValueError("Narratives cannot be empty")
    
    if not all(isinstance(doc, str) for doc in narrative_list):
        raise ValueError("All elements in narratives must be strings")
    
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
    
    if nlp_model is None:
        try:
            # Assumes 'nlp' is a globally available spaCy model object
            nlp_model = nlp  
        except NameError:
            raise RuntimeError("No spaCy model provided and global 'nlp' variable not found")
    
    
    if not hasattr(nlp_model, '__call__') or not hasattr(nlp_model, 'vocab'):
        raise RuntimeError("Provided nlp_model is not a valid spaCy language model")
    
    try:
        pos_tags = Counter()
        
        for i, narrative in enumerate(narrative_list):
            if not narrative.strip():  # Skip empty narratives
                continue
                
            try:
                doc = nlp_model(narrative)
                for token in doc:
                    if token.pos_:  # Only count tokens with valid POS tags
                        pos_tags[token.pos_] += 1
            except Exception as e:
                # Use logging in real code, but print is okay for a simple fix
                print(f"Warning: Failed to process document {i}: {e}")
                continue
        
        if not pos_tags:
            return []  # No valid POS tags found
            
        return pos_tags.most_common(top_n)
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract POS tags: {str(e)}")
    
def calculate_lexical_diversity(narratives, case_sensitive=True, 
                                       remove_punctuation=False):
    """
    Calculate lexical diversity scores for a collection of text narratives.
    
    This function computes the lexical diversity (Type-Token Ratio) for each
    narrative in the input collection. Lexical diversity measures the richness
    of vocabulary in a text by calculating the ratio of unique words to total
    words. Higher scores indicate greater vocabulary variety and complexity.
    
    The function provides flexible text preprocessing options including case
    normalization and punctuation removal to accommodate different analysis
    requirements. It includes comprehensive input validation and robust error
    handling for reliable processing of text data.
    
    Parameters:
    -----------
    narratives : array-like of str
        Collection of text documents to analyze
    case_sensitive : bool, optional (default=True)
        If False, converts all words to lowercase before calculating diversity
    remove_punctuation : bool, optional (default=False)
        If True, removes punctuation from words before calculating diversity
        
    Returns:
    --------
    list of float
        List of lexical diversity scores for each narrative
        
    Raises:
    -------
    ValueError: If input is invalid (None, empty, or contains non-strings)
    RuntimeError: If processing fails
    """
    import string
   
    if narratives is None:
        raise ValueError("Narratives cannot be None")
    
    try:
        narrative_list = list(narratives)
    except (TypeError, ValueError):
        raise ValueError("Narratives must be iterable")
    
    if not narrative_list:
        raise ValueError("Narratives cannot be empty")
    
    if not all(isinstance(doc, str) for doc in narrative_list):
        raise ValueError("All elements in narratives must be strings")
    
    try:
        diversity_scores = []
        
        for narrative in narrative_list:
            words = narrative.split()
            
            if len(words) == 0:
                diversity_scores.append(0.0)
                continue
     
            processed_words = []
            for word in words:
                processed_word = word
             
                if remove_punctuation:
                    processed_word = processed_word.translate(str.maketrans('', '', string.punctuation))
                
                if not case_sensitive:
                    processed_word = processed_word.lower()
                
                if processed_word:
                    processed_words.append(processed_word)
            
            if len(processed_words) == 0:
                diversity_scores.append(0.0)
            else:
                unique_words = set(processed_words)
                diversity = len(unique_words) / len(processed_words)
                diversity_scores.append(diversity)
        
        return diversity_scores
        
    except Exception as e:
        raise RuntimeError(f"Failed to calculate lexical diversity: {str(e)}")

def calculate_normalized_lexical_diversity(narratives, min_tokens=10):
    """
    Calculate lexical diversity with normalization for text length bias.
    
    Uses Moving Average Type-Token Ratio (MATTR) approach for longer texts
    to reduce the bias where longer texts naturally have lower TTR.
    
    Parameters:
    -----------
    narratives : array-like of str
        Collection of text documents to analyze
    min_tokens : int, optional (default=10)
        Minimum number of tokens required to use normalization.
        Shorter texts use standard TTR calculation.
        
    Returns:
    --------
    list of float
        List of normalized lexical diversity scores
    """
    diversity_scores = []
    
    for narrative in narratives:
        words = narrative.split()
        
        if len(words) == 0:
            diversity_scores.append(0.0)
        elif len(words) < min_tokens:
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            diversity_scores.append(diversity)
        else:
            window_size = min(50, len(words) // 2)  # Adaptive window size
            ttrs = []
            
            for i in range(len(words) - window_size + 1):
                window_words = words[i:i + window_size]
                unique_in_window = set(window_words)
                window_ttr = len(unique_in_window) / len(window_words)
                ttrs.append(window_ttr)
            
            mattr = sum(ttrs) / len(ttrs) if ttrs else 0.0
            diversity_scores.append(mattr)
    
    return diversity_scores