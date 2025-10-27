# preprocessing.py
import string
import pandas as pd
import logging
import re
from tqdm import tqdm
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from utilities import (
    preprocess_text, 
    handle_punctuation, 
    remove_num, 
    remove_words_dict,
    calculate_idf,
    spacy_tokenizer
)
_cached_models = {}

def pre_processing_routine(SDRS, text_column="Narrative_long"):
    """
    Standard preprocessing routine for narratives.
    
    Applies text preprocessing steps to narratives in the SDRS dataframe,
    including text normalization, punctuation handling, and optional number removal.
    
    Parameters
    ----------
    SDRS : pandas.DataFrame
        DataFrame containing text narratives to process.
    text_column : str, optional
        Name of the column containing text to process (default: "Narrative_long").
    
    Returns
    -------
    tuple of (list, list)
        processed : list
            Preprocessed narratives with numbers retained (punctuation handled).
        processed_numberless : list
            Preprocessed narratives with numbers removed and re-preprocessed.
    
    Raises
    ------
    ValueError
        If SDRS is not a DataFrame or text_column doesn't exist.
    
    Notes
    -----
    Processing pipeline:
    1. Initial text preprocessing (preprocess_text)
    2. Punctuation handling (handle_punctuation)
    3. For numberless version: number removal (remove_num) + re-preprocessing
    """
    # Input validation
    if not isinstance(SDRS, pd.DataFrame):
        raise ValueError("SDRS must be a pandas DataFrame")
    
    if text_column not in SDRS.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. "
                        f"Available columns: {list(SDRS.columns)}")
    
    processed = []
    processed_numberless = []
    for narrative in SDRS[text_column]:
        pre_proc_narr = preprocess_text(narrative)
        punc_narr = handle_punctuation(pre_proc_narr)
        num_narr = remove_num(punc_narr)
        num_narr = preprocess_text(num_narr)  # Re-preprocessing after number removal
        processed.append(punc_narr)
        processed_numberless.append(num_narr)
    return processed, processed_numberless

def remove_stopwords_corpus(corpus, stopwords):
    """
    Remove stopwords from an entire corpus of text documents.
    
    Applies stopword removal to each document in the corpus using the
    remove_words_dict function. Skips null/NaN values and handles errors
    gracefully by preserving the original text if processing fails.
    
    Parameters
    ----------
    corpus : list of str
        List of text documents to process.
    stopwords : set, list, or dict
        Collection of stopwords to remove from each document.
        Should be compatible with remove_words_dict function.
    
    Returns
    -------
    list of str
        Corpus with stopwords removed from each document. Null values are
        replaced with empty strings, and documents that fail processing
        are returned unchanged.
    
    Raises
    ------
    TypeError
        If corpus is not iterable.
    ValueError
        If stopwords is None or empty.
    """
    # Input validation
    if not hasattr(corpus, '__iter__') or isinstance(corpus, str):
        raise TypeError("corpus must be an iterable (list, tuple, etc.), not a string or non-iterable")
    
    if stopwords is None or (hasattr(stopwords, '__len__') and len(stopwords) == 0):
        raise ValueError("stopwords cannot be None or empty")
    
    processed = []
    for text in corpus:
        # Handle null/NaN values
        if pd.notna(text) and text is not None:
            try:
                # Attempt to remove stopwords
                cleaned_text = remove_words_dict(text, stopwords)
                processed.append(cleaned_text)
            except Exception as e:
                # If processing fails, keep original text and optionally log warning
                print(f"Warning: Failed to process document, keeping original. Error: {e}")
                processed.append(text)
        else:
            # Replace null values with empty strings
            processed.append("")
    
    return processed

def define_stopwords(SDRS, 
                    text_column="Narrative_long",
                    custom_stop_words=None, 
                    custom_keep_words=None, 
                    low_idf=True, 
                    idf_threshold=1.4):
    """
    Define stopwords by combining default stopwords, custom domain-specific words,
    and optionally low IDF words from the corpus.
    
    Parameters
    ----------
    SDRS : pandas.DataFrame
        DataFrame containing processed narratives for IDF calculation.
    text_column : str, optional
        Name of the column containing text for IDF calculation (default: "Narrative_long").
    custom_stop_words : list of str or None, optional
        Domain-specific stop words to add. If None, defaults to aviation-related terms.
    custom_keep_words : list of str or None, optional
        Words to explicitly exclude from the final stopword list. If None, defaults to [].
    low_idf : bool, optional
        Whether to include low IDF words as stopwords (default: True).
    idf_threshold : float, optional
        IDF threshold below which words are considered stopwords (default: 1.4).
    
    Returns
    -------
    list of str
        Combined list of stopwords with duplicates removed and keep words excluded.
    
    Raises
    ------
    ValueError
        If SDRS is None or empty when low_idf is True, or if text_column doesn't exist.
    TypeError
        If custom_stop_words or custom_keep_words are not iterable.
    
    Notes
    -----
    The function combines three sources of stopwords:
    1. Default stopwords from STOP_WORDS
    2. Custom domain-specific stopwords
    3. Low IDF words from the corpus (if low_idf=True)
    
    Words in custom_keep_words are removed from the final stopword list.
    """
    # Initialize mutable defaults inside function
    if custom_stop_words is None:
        custom_stop_words = ['pilot', 'attendant', 'aircraft', 'flight', 
                            'captain', 'officer', 'first officer']
    
    if custom_keep_words is None:
        custom_keep_words = []
    
    # Validate custom_stop_words and custom_keep_words are iterable
    if not hasattr(custom_stop_words, '__iter__') or isinstance(custom_stop_words, str):
        raise TypeError("custom_stop_words must be an iterable (list, set, etc.), not a string or non-iterable")
    
    if not hasattr(custom_keep_words, '__iter__') or isinstance(custom_keep_words, str):
        raise TypeError("custom_keep_words must be an iterable (list, set, etc.), not a string or non-iterable")
    
    # Input validation for SDRS when low_idf is True
    if low_idf:
        if SDRS is None or (hasattr(SDRS, '__len__') and len(SDRS) == 0):
            raise ValueError("SDRS cannot be None or empty when low_idf is True")
        
        if not isinstance(SDRS, pd.DataFrame):
            raise ValueError("SDRS must be a pandas DataFrame")
        
        if text_column not in SDRS.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame. "
                            f"Available columns: {list(SDRS.columns)}")
    
    # Convert to set for efficiency (O(1) lookups vs O(n) for lists)
    stopwords_set = set(STOP_WORDS)
    
    # Add custom stopwords
    stopwords_set.update(custom_stop_words)
    
    # Add low IDF stopwords if requested with error handling
    if low_idf:
        try:
            
            narratives_list = SDRS[text_column].dropna().astype(str).tolist()
            
            # Check if we have valid data
            if not narratives_list:
                print(f"Warning: No valid data in column '{text_column}' for IDF calculation. Skipping low IDF stopwords.")
            else:
                # Call calculate_idf with the list of narratives (not DataFrame)
                idf_results = calculate_idf(narratives_list, text_column=text_column)
                
                # Filter words below threshold
                low_idf_stopwords = [word for (IDF, word) in idf_results 
                                    if IDF < idf_threshold]
                
                print(f"Found {len(low_idf_stopwords)} low-IDF words (threshold: {idf_threshold})")
                stopwords_set.update(low_idf_stopwords)
                
        except Exception as e:
            print(f"Warning: Failed to calculate IDF, skipping low IDF stopwords. Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Remove words we want to keep (convert to set for O(1) lookup)
    custom_keep_set = set(custom_keep_words)
    stopwords_set.difference_update(custom_keep_set)
    
    # Convert back to list and return
    return list(stopwords_set)

def load_spacy_model(disable_pipes=None, use_cache=True):
    """
    Load and configure the spaCy English language model.
    
    Loads the 'en_core_web_lg' spaCy model and optionally disables specified
    pipeline components to improve processing speed when certain features
    are not needed. Models are cached by default to avoid expensive reloading.
    
    Parameters
    ----------
    disable_pipes : list of str or None, optional
        List of pipeline component names to disable. If None, defaults to ['ner'].
        Common pipes include: 'tok2vec', 'tagger', 'parser', 'ner', 'lemmatizer',
        'attribute_ruler', 'senter'.
    use_cache : bool, optional
        Whether to use cached model if available (default: True). Set to False
        to force reload the model.
    
    Returns
    -------
    spacy.language.Language
        Configured spaCy language model with specified pipes disabled.
    
    Raises
    ------
    OSError
        If the 'en_core_web_lg' model is not installed.
    ValueError
        If disable_pipes contains invalid pipe names or is not iterable.
    
    Notes
    -----
    The 'en_core_web_lg' model must be installed before using this function:
        python -m spacy download en_core_web_lg
    
    Disabling unnecessary pipes can significantly improve processing speed.
    For example, if you only need tokenization and lemmatization, you can
    disable 'ner', 'parser', and 'tagger'.
    
    """
    # Initialize default
    if disable_pipes is None:
        disable_pipes = ['ner']
    
    # Validate input type
    if not hasattr(disable_pipes, '__iter__') or isinstance(disable_pipes, str):
        raise ValueError("disable_pipes must be an iterable (list, set, etc.), not a string or non-iterable")
    
    # Convert to tuple for use as dict key (lists aren't hashable)
    disable_pipes_tuple = tuple(sorted(disable_pipes))
    
    # Check cache if enabled
    if use_cache and disable_pipes_tuple in _cached_models:
        logging.info(f"Using cached spaCy model with disabled pipes: {list(disable_pipes_tuple)}")
        return _cached_models[disable_pipes_tuple]
    
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError as e:
        raise OSError(
            "Could not load 'en_core_web_lg' model. "
            "Please install it with: python -m spacy download en_core_web_lg"
        ) from e
    
    # Validate pipe names before disabling
    if disable_pipes:
        available_pipes = nlp.pipe_names
        invalid_pipes = [pipe for pipe in disable_pipes if pipe not in available_pipes]
        
        if invalid_pipes:
            raise ValueError(
                f"Invalid pipe names: {invalid_pipes}. "
                f"Available pipes are: {available_pipes}"
            )
        
        # Disable specified pipes
        try:
            nlp.disable_pipes(*disable_pipes)
            logging.info(f"Disabled spaCy pipes: {list(disable_pipes)}")
        except Exception as e:
            raise RuntimeError(f"Failed to disable pipes: {list(disable_pipes)}. Error: {e}") from e
    
    # Log active pipes
    logging.info(f"Active spaCy pipes: {nlp.pipe_names}")
    
    # Cache the model if caching is enabled
    if use_cache:
        _cached_models[disable_pipes_tuple] = nlp
        logging.info(f"Cached spaCy model with configuration: {list(disable_pipes_tuple)}")
    
    return nlp


def clear_spacy_cache():
    """
    Clear all cached spaCy models from memory.
    
    Use this function to free up memory (~560MB per cached model) if you're
    done processing or need to switch to a different model configuration
    frequently.

    Returns
    -------
    int
        Number of cached models cleared.
    """
    global _cached_models
    count = len(_cached_models)
    _cached_models.clear()
    logging.info(f"Cleared {count} cached spaCy model(s) from memory")
    return count

def process_with_spacy(SDRS, text_column="processed_stopword_Narr", spacy_tokenizer=None, 
                       batch_size=1000, n_process=1):
    """
    Apply spaCy processing to a DataFrame column using efficient batch processing.
    
    Uses spaCy's nlp.pipe() method for efficient batch processing, which is
    10-100x faster than applying spaCy to individual documents.
    
    Parameters
    ----------
    SDRS : pandas.DataFrame
        DataFrame containing text to process.
    text_column : str, optional
        Name of column containing text to process (default: "processed_stopword_Narr").
    spacy_tokenizer : callable or None, optional
        Function to apply to each spaCy Doc object. If None, returns the raw Doc objects.
        The function should take a spaCy Doc as input and return the desired output.
    batch_size : int, optional
        Number of texts to process in each batch (default: 1000).
        Larger batches are faster but use more memory.
    n_process : int, optional
        Number of parallel processes to use (default: 1).
        Set to -1 to use all available CPUs. Note: multiprocessing only works
        with batch_size and may have overhead for small datasets.
    
    Returns
    -------
    pandas.Series
        Series with processed text, same length as input DataFrame.
    
    Raises
    ------
    ValueError
        If SDRS is not a DataFrame or text_column doesn't exist.
    TypeError
        If spacy_tokenizer is not callable.
    
    Notes
    -----
    This function uses spaCy's nlp.pipe() for efficient batch processing.
    Null/NaN values in the text column are handled gracefully and returned as None.
    
    The spaCy model must be loaded separately (via load_spacy_model) before
    the spacy_tokenizer function is defined.
    
    Examples
    --------
    >>> # Define a tokenizer function
    >>> nlp = load_spacy_model(disable_pipes=['ner'])
    >>> def my_tokenizer(doc):
    ...     return [token.lemma_ for token in doc]
    >>> 
    >>> # Process DataFrame
    >>> df['processed'] = process_with_spacy(df, 'text_column', my_tokenizer)
    >>> 
    >>> # Or get raw Doc objects
    >>> df['docs'] = process_with_spacy(df, 'text_column')
    """
    # Input validation - check DataFrame
    if not isinstance(SDRS, pd.DataFrame):
        raise ValueError("SDRS must be a pandas DataFrame")
    
    # Check if column exists
    if text_column not in SDRS.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. "
                        f"Available columns: {list(SDRS.columns)}")
    
    # Validate spacy_tokenizer if provided
    if spacy_tokenizer is not None and not callable(spacy_tokenizer):
        raise TypeError("spacy_tokenizer must be a callable function or None")
    
    # Get the text series
    text_series = SDRS[text_column]
    
    # Handle empty DataFrame
    if len(text_series) == 0:
        logging.warning("Empty DataFrame provided, returning empty Series")
        return pd.Series([], dtype=object)
    
    # Load spaCy model (will use cached version if available)
    nlp = load_spacy_model()
    
    # Separate null and non-null indices
    non_null_mask = text_series.notna()
    non_null_indices = text_series[non_null_mask].index
    non_null_texts = text_series[non_null_mask].tolist()
    
    # Initialize results list with None for all positions
    results = [None] * len(text_series)
    
    # Process non-null texts with progress bar
    if non_null_texts:
        logging.info(f"Processing {len(non_null_texts)} documents with spaCy (batch_size={batch_size})")
        
        try:
            # Use tqdm for progress tracking with nlp.pipe
            processed_docs = []
            for doc in tqdm(nlp.pipe(non_null_texts, batch_size=batch_size, n_process=n_process),
                           total=len(non_null_texts),
                           desc="Processing with spaCy"):
                # Apply tokenizer function if provided, otherwise keep Doc object
                if spacy_tokenizer is not None:
                    try:
                        processed_docs.append(spacy_tokenizer(doc))
                    except Exception as e:
                        logging.warning(f"Error processing document with spacy_tokenizer: {e}")
                        processed_docs.append(None)
                else:
                    processed_docs.append(doc)
            
            # Place processed docs back in their original positions
            for idx, result in zip(non_null_indices, processed_docs):
                results[SDRS.index.get_loc(idx)] = result
                
        except Exception as e:
            raise RuntimeError(f"Failed to process texts with spaCy: {e}") from e
    else:
        logging.warning("No non-null texts to process")
    
    # Convert to Series with original index
    return pd.Series(results, index=SDRS.index)

def expand_contractions(text):
    """
    Expand common English contractions in text.
    
    Replaces contracted forms (e.g., "don't", "can't") with their expanded
    forms (e.g., "do not", "cannot") using word boundary matching to avoid
    false positives.
    
    Parameters
    ----------
    text : str
        Input text containing contractions to expand.
    
    Returns
    -------
    str
        Text with contractions expanded. Returns empty string if input is None/NaN.
    
    Limitations
    -----------
    - Does not handle context-dependent ambiguities
    - May not cover all regional or informal contractions
    """
    # Initialize contractions dict only once using function attribute
    if not hasattr(expand_contractions, '_contractions_dict'):
        expand_contractions._contractions_dict = {
            # Common negations
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "mightn't": "might not",
            "mustn't": "must not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            
            # "to be" contractions
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",  # Note: could also be "he has" - context dependent
            "she's": "she is",  # Note: could also be "she has" - context dependent
            "it's": "it is",  # Note: could also be "it has" - context dependent
            "we're": "we are",
            "they're": "they are",
            "that's": "that is",  # Note: could also be "that has" - context dependent
            "who's": "who is",  # Note: could also be "who has" - context dependent
            "what's": "what is",  # Note: could also be "what has" - context dependent
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
            "there's": "there is",
            "here's": "here is",
            
            # "have" contractions
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            
            # "will" contractions
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "it'll": "it will",
            "that'll": "that will",
            
            # "would/had" contractions (ambiguous - defaulting to "would")
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "it'd": "it would",
            "that'd": "that would"
        }
    
    # Use the cached dictionary
    contractions_dict = expand_contractions._contractions_dict
    
    # Input validation - handle None, NaN, and non-string types
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            logging.warning(f"Could not convert input to string: {e}")
            return ""
    
    # Handle empty string
    if not text:
        return ""
    
    # Expand contractions using regex with word boundaries
    try:
        for contraction, expansion in contractions_dict.items():
            # Use word boundaries to avoid partial matches
            # Example: won't match "don't" in "donuts"
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text)
    except Exception as e:
        logging.error(f"Error expanding contractions: {e}")
        # Return original text if expansion fails
        return text
    
    return text

def remove_special_characters(text, keep_chars=".,!?", preserve_unicode=True):
    """
    Remove special characters while optionally keeping some punctuation.
    
    Removes all characters except alphanumeric characters, whitespace, and 
    specified punctuation to keep. Can preserve Unicode letters and digits
    or restrict to ASCII only.
    
    Parameters
    ----------
    text : str
        Input text to clean.
    keep_chars : str or None, optional
        String of punctuation characters to preserve (default: ".,!?").
        Set to empty string "" to remove all punctuation.
        Set to None to use default.
    preserve_unicode : bool, optional
        If True (default), preserves Unicode letters and digits (café, 日本語).
        If False, only preserves ASCII alphanumeric characters (a-z, A-Z, 0-9).
    
    Returns
    -------
    str
        Text with special characters removed. Returns empty string if input is None/NaN.
    """
    # Initialize pattern cache using function attribute
    if not hasattr(remove_special_characters, '_pattern_cache'):
        remove_special_characters._pattern_cache = {}
    
    # Input validation - handle None, NaN, and non-string types
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            logging.warning(f"Could not convert input to string: {e}")
            return ""
    
    # Handle empty string
    if not text:
        return ""
    
    # Handle None for keep_chars
    if keep_chars is None:
        keep_chars = ".,!?"
    
    # Validate keep_chars is a string
    if not isinstance(keep_chars, str):
        try:
            keep_chars = str(keep_chars)
        except Exception as e:
            logging.warning(f"Could not convert keep_chars to string: {e}, using default")
            keep_chars = ".,!?"
    
    # Create cache key including preserve_unicode setting
    cache_key = (keep_chars, preserve_unicode)
    
    # Check cache for compiled pattern
    if cache_key not in remove_special_characters._pattern_cache:
        try:
            # Escape keep_chars for safe use in regex character class
            escaped_chars = re.escape(keep_chars)
            
            # Create pattern based on Unicode preference
            if preserve_unicode:
                # \w matches Unicode letters, digits, and underscore
                # We need to explicitly add underscore to removal if not in keep_chars
                # Pattern: keep Unicode word chars (letters/digits), whitespace, and keep_chars
                pattern = f"[^\\w\\s{escaped_chars}]"
            else:
                # ASCII only: keep a-z, A-Z, 0-9, whitespace, and keep_chars
                pattern = f"[^a-zA-Z0-9\\s{escaped_chars}]"
            
            # Compile and cache the pattern with UNICODE flag for proper Unicode handling
            remove_special_characters._pattern_cache[cache_key] = re.compile(pattern, re.UNICODE)
            
        except Exception as e:
            logging.error(f"Error compiling regex pattern with keep_chars='{keep_chars}': {e}")
            # Fallback to default pattern
            if preserve_unicode:
                remove_special_characters._pattern_cache[cache_key] = re.compile(r"[^\w\s]", re.UNICODE)
            else:
                remove_special_characters._pattern_cache[cache_key] = re.compile(r"[^a-zA-Z0-9\s]")
    
    # Apply cached compiled pattern
    try:
        compiled_pattern = remove_special_characters._pattern_cache[cache_key]
        result = compiled_pattern.sub('', text)
        
        # Note: \w includes underscore, so we need to remove it if not in keep_chars
        if preserve_unicode and '_' not in keep_chars:
            result = result.replace('_', '')
        
        return result
        
    except Exception as e:
        logging.error(f"Error applying regex substitution: {e}")
        # Return original text if substitution fails
        return text

def normalize_whitespace(text):
    """
    Normalize all whitespace characters to single spaces.
    
    Replaces all sequences of whitespace characters (spaces, tabs, newlines,
    carriage returns, etc.) with a single space and removes leading/trailing
    whitespace.
    
    Parameters
    ----------
    text : str
        Input text containing whitespace to normalize.
    
    Returns
    -------
    str
        Text with normalized whitespace. Returns empty string if input is None/NaN.
     
    """
    # Input validation - handle None, NaN, and non-string types
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            logging.warning(f"Could not convert input to string: {e}")
            return ""
    
    # Handle empty string
    if not text:
        return ""
    
    # Normalize whitespace
    try:
        return ' '.join(text.split())
    except Exception as e:
        logging.error(f"Error normalizing whitespace: {e}")
        # Return original text if normalization fails
        return text

def clean_and_tokenize_pipeline(text, nlp, keep_pos=None):
    """
    Complete text cleaning and tokenization pipeline.
    
    Applies a series of text preprocessing steps followed by spaCy-based
    tokenization, lemmatization, and POS filtering to produce clean tokens.
    
    Parameters
    ----------
    text : str
        Input text to process.
    nlp : spacy.language.Language
        Loaded spaCy model (from load_spacy_model()).
    keep_pos : list of str or None, optional
        List of POS (Part-of-Speech) tags to keep. If None, defaults to
        ['NOUN', 'VERB', 'ADJ', 'ADV']. Common POS tags include:
        'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM', etc.
    
    Returns
    -------
    str
        Space-separated string of cleaned and lemmatized tokens.
        Returns empty string if no tokens match criteria or input is None/NaN.
    
    Notes
    -----
    Processing pipeline:
    1. Lowercase text
    2. Expand contractions (don't → do not)
    3. Remove special characters (keeps basic punctuation by default)
    4. Normalize whitespace
    5. spaCy processing (tokenization, POS tagging, lemmatization)
    6. Filter tokens by:
       - POS tag (must be in keep_pos)
       - Not a stopword
       - Not punctuation
       - Length > 2 characters
    7. Return space-separated lemmatized tokens
  
    """
    # Initialize default for keep_pos
    if keep_pos is None:
        keep_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
    
    # Input validation - handle None, NaN, and non-string types
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            logging.warning(f"Could not convert input to string: {e}")
            return ""
    
    # Handle empty string
    if not text or not text.strip():
        return ""
    
    # Validate nlp model
    if nlp is None:
        raise ValueError("nlp model cannot be None. Load a spaCy model using load_spacy_model()")
    
    # Validate keep_pos
    if not hasattr(keep_pos, '__iter__') or isinstance(keep_pos, str):
        raise TypeError("keep_pos must be an iterable (list, set, tuple), not a string or non-iterable")
    
    # Convert keep_pos to set for O(1) lookup efficiency
    keep_pos_set = set(keep_pos)
    
    if not keep_pos_set:
        logging.warning("keep_pos is empty, no tokens will be kept")
        return ""
    
    try:
        # Basic cleaning pipeline
        text = text.lower()
        text = expand_contractions(text)
        text = remove_special_characters(text)
        text = normalize_whitespace(text)
        
        # Handle case where cleaning results in empty string
        if not text or not text.strip():
            return ""
        
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        return ""
    
    # spaCy processing
    try:
        doc = nlp(text)
        
        tokens = [
            token.lemma_.lower()
            for token in doc 
            if (token.pos_ in keep_pos_set and  # O(1) lookup with set
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2)
        ]
        
        # Handle empty result
        if not tokens:
            logging.debug("No tokens matched the filtering criteria")
            return ""
        
        return ' '.join(tokens)
        
    except Exception as e:
        logging.error(f"Error during spaCy processing: {e}")
        return ""

def batch_preprocess(SDRS, text_column="Narrative_long", nlp=None, keep_pos=None, 
                     custom_stop_words=None, custom_keep_words=None, 
                     low_idf=True, idf_threshold=1.4, 
                     copy_dataframe=True):
    """
    Complete batch preprocessing pipeline for a DataFrame.
    
    Applies a comprehensive text preprocessing pipeline including cleaning,
    stopword removal, and spaCy-based tokenization to produce final processed text.
    
    Parameters
    ----------
    SDRS : pandas.DataFrame
        DataFrame containing text data to process.
    text_column : str, optional
        Name of the column containing raw text to process (default: "Narrative_long").
    nlp : spacy.language.Language or None, optional
        Pre-loaded spaCy model. If None, loads model automatically.
    keep_pos : list of str or None, optional
        POS tags to keep in final processing. If None, uses default ['NOUN', 'VERB', 'ADJ', 'ADV'].
    custom_stop_words : list of str or None, optional
        Custom stopwords to add. If None, uses default domain-specific words.
    custom_keep_words : list of str or None, optional
        Words to exclude from stopword removal. If None, uses empty list.
    low_idf : bool, optional
        Whether to include low IDF words as stopwords (default: True).
    idf_threshold : float, optional
        IDF threshold for low IDF stopwords (default: 1.4).
    copy_dataframe : bool, optional
        If True (default), works on a copy of the DataFrame and returns it.
        If False, modifies the input DataFrame in place and returns it.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns:
        - 'processed': Text after initial preprocessing (with numbers)
        - 'processed_num_Narr': Text after initial preprocessing (numbers removed)
        - 'processed_stopword_Narr': Text after stopword removal
        - 'Final_Processed': Final cleaned and tokenized text (space-separated lemmas)
    
    Raises
    ------
    ValueError
        If SDRS is not a DataFrame or text_column doesn't exist.
    
    """
    # Input validation
    if not isinstance(SDRS, pd.DataFrame):
        raise ValueError("SDRS must be a pandas DataFrame")
    
    if text_column not in SDRS.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. "
                        f"Available columns: {list(SDRS.columns)}")
    
    if len(SDRS) == 0:
        logging.warning("Empty DataFrame provided")
        return SDRS.copy() if copy_dataframe else SDRS
    
    # Work on a copy if requested to avoid modifying original
    if copy_dataframe:
        SDRS = SDRS.copy()
        logging.info("Working on a copy of the DataFrame")
    else:
        logging.info("Modifying DataFrame in place")
    
    # Load spaCy model if not provided
    if nlp is None:
        logging.info("Loading spaCy model...")
        try:
            nlp = load_spacy_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load spaCy model: {e}") from e
    
    try:
        # Step 1: Initial preprocessing
        logging.info("Step 1/4: Running initial preprocessing...")
        SDRS["processed"], SDRS["processed_num_Narr"] = pre_processing_routine(SDRS, text_column=text_column)
        
        # Validate that columns were created
        if "processed_num_Narr" not in SDRS.columns or SDRS["processed_num_Narr"].isna().all():
            raise RuntimeError("Initial preprocessing failed to produce valid output")
        
    except Exception as e:
        logging.error(f"Error in initial preprocessing: {e}")
        raise RuntimeError(f"Initial preprocessing failed: {e}") from e
    
    try:
        # Step 2: Define stopwords
        logging.info("Step 2/4: Defining stopwords...")
        stopwords = define_stopwords(
            SDRS, 
            text_column="processed_num_Narr",
            custom_stop_words=custom_stop_words,
            custom_keep_words=custom_keep_words,
            low_idf=low_idf,
            idf_threshold=idf_threshold
        )
        logging.info(f"Defined {len(stopwords)} stopwords")
        
    except Exception as e:
        logging.error(f"Error defining stopwords: {e}")
        raise RuntimeError(f"Stopword definition failed: {e}") from e
    
    try:
        # Step 3: Remove stopwords
        logging.info("Step 3/4: Removing stopwords...")
        SDRS["processed_stopword_Narr"] = remove_stopwords_corpus(
            SDRS["processed_num_Narr"].tolist(), stopwords)
        
        # Validate stopword removal
        if SDRS["processed_stopword_Narr"].isna().all():
            logging.warning("Stopword removal resulted in all null values")
        
    except Exception as e:
        logging.error(f"Error removing stopwords: {e}")
        raise RuntimeError(f"Stopword removal failed: {e}") from e
    
    try:
        # Step 4: Final processing with spaCy in batch mode
        logging.info("Step 4/4: Running final spaCy processing and cleaning...")
        
        # Get texts to process
        texts_to_process = SDRS["processed_stopword_Narr"].tolist()
        
        # Initialize results list
        final_processed = []
        
        # Process in batches using nlp.pipe for efficiency
        batch_size = 1000
        total_docs = len(texts_to_process)
        
        for doc in tqdm(nlp.pipe(texts_to_process, batch_size=batch_size), 
                       total=total_docs,
                       desc="Final processing"):
            try:
                # Initialize keep_pos if None
                pos_tags = keep_pos if keep_pos is not None else ['NOUN', 'VERB', 'ADJ', 'ADV']
                keep_pos_set = set(pos_tags)
                
                # Extract and filter tokens directly from doc
                # This avoids redundant cleaning since text is already preprocessed
                tokens = [
                    token.lemma_.lower()
                    for token in doc 
                    if (token.pos_ in keep_pos_set and  
                        not token.is_stop and 
                        not token.is_punct and 
                        len(token.text) > 2)
                ]
                
                final_processed.append(' '.join(tokens) if tokens else "")
                
            except Exception as e:
                logging.warning(f"Error processing document: {e}")
                final_processed.append("")
        
        # Assign to DataFrame
        SDRS["Final_Processed"] = final_processed
        
        # Log statistics
        non_empty = sum(1 for text in final_processed if text)
        logging.info(f"Processing complete: {non_empty}/{total_docs} documents produced non-empty results")
        
    except Exception as e:
        logging.error(f"Error in final spaCy processing: {e}")
        raise RuntimeError(f"Final processing failed: {e}") from e
    
    return SDRS