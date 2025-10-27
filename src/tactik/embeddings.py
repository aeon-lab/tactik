# embeddings.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from umap import UMAP
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from gensim import downloader as api 
from gensim.models import Word2Vec
import numpy as np
from .utilities import get_number_unique_words
import torch


def vectorize_text(text, max_features=None):
    """
    TF-IDF vectorization of text documents with vocabulary size reporting.
    
    This function converts a collection of text documents into a TF-IDF 
    (Term Frequency-Inverse Document Frequency) matrix representation. It uses
    sklearn's TfidfVectorizer to transform raw text into a numerical format
    suitable for machine learning algorithms.
    
    The function efficiently computes the vocabulary size from the fitted
    vectorizer without requiring additional passes through the data.
    
    Parameters:
    -----------
    text : array-like of str
        Collection of text documents to vectorize. Each element should be a 
        string representing a document. Can be a list, numpy array, pandas 
        Series, or any iterable containing text documents.
        
    max_features : int, optional (default=None)
        Maximum number of features (vocabulary size) to use. If specified, 
        only the top `max_features` ordered by term frequency across the 
        corpus will be used. If None, all unique words will be used as features.
        Common values: 1000, 5000, 10000 for dimensionality reduction.
        
    Returns:
    --------
    tfidf_matrix : scipy.sparse matrix of shape (n_documents, n_features)
        TF-IDF weighted document-term matrix. Each row represents a document
        and each column represents a term in the vocabulary. Values are the
        TF-IDF scores.
        
    vectorizer : TfidfVectorizer
        The fitted TfidfVectorizer object. This can be used to transform
        additional documents using the same vocabulary with:
        `vectorizer.transform(new_documents)`
        
    vocab_size : int
        The actual number of features (unique terms) in the vocabulary after
        fitting. If max_features was specified and there are more unique words,
        this will equal max_features. Otherwise, it equals the total number of
        unique words found in the corpus.
        
    Raises:
    -------
    ValueError
        If text is None, empty, or contains non-string elements.
        
    Examples:
    ---------
    >>> documents = [
    ...     "This is the first document.",
    ...     "This document is the second document.",
    ...     "And this is the third one."
    ... ]
    >>> tfidf_matrix, vectorizer, vocab_size = vectorize_text(documents)
    >>> print(f"Vocabulary size: {vocab_size}")
    >>> print(f"Matrix shape: {tfidf_matrix.shape}")
    
    >>> # Limit to top 5 features
    >>> tfidf_matrix, vectorizer, vocab_size = vectorize_text(documents, max_features=5)
    >>> print(f"Limited vocabulary size: {vocab_size}")
    

    """
    # Input validation
    if text is None:
        raise ValueError("Text cannot be None")
    
    try:
        text_list = list(text)
    except (TypeError, ValueError):
        raise ValueError("Text must be iterable")
    
    if not text_list:
        raise ValueError("Text cannot be empty")
    
    if not all(isinstance(doc, str) for doc in text_list):
        raise ValueError("All elements in text must be strings")
    
    # Fit the vectorizer and transform the text
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(text_list)
        
        # Get vocabulary size from the fitted vectorizer (no extra computation)
        vocab_size = len(vectorizer.get_feature_names_out())
        
        return tfidf_matrix, vectorizer, vocab_size
        
    except Exception as e:
        raise RuntimeError(f"Failed to vectorize text: {str(e)}")
def get_umap_embeddings(X, n_components=5, n_neighbors=10, min_dist=0.0, 
                        metric='cosine', random_state=None, verbose=False):
    """
    Generate UMAP embeddings for dimensionality reduction of text data.
    
    This function applies UMAP (Uniform Manifold Approximation and Projection)
    to reduce high-dimensional text features (such as TF-IDF vectors) into a
    lower-dimensional embedding space while preserving both local and global
    structure of the data.
    
    Parameters:
    -----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        Input features to embed. Typically a TF-IDF matrix, count matrix, or
        other numerical representation of text. Can be dense numpy array or
        scipy sparse matrix.
        
    n_components : int, optional (default=5)
        Dimension of the output embedding space. Common values:
        - 2 or 3 for visualization
        - 5-50 for downstream machine learning tasks
        Must be less than the number of features in X.
        
    n_neighbors : int, optional (default=10)
        Number of neighboring points used in local approximations of manifold
        structure. Larger values capture more global structure, smaller values
        preserve more local structure. Must be less than the number of samples.
        Typical range: 5-50.
        
    min_dist : float, optional (default=0.0)
        Minimum distance between points in the embedded space. Controls how
        tightly UMAP packs points together:
        - 0.0: Allows points to be packed closely (good for clustering)
        - 0.1-0.5: More spread out (better for visualization)
        Range: 0.0 to 1.0.
        
    metric : str or callable, optional (default='cosine')
        Distance metric to use for computing distances in the input space.
        Common options for text:
        - 'cosine': Cosine distance (recommended for TF-IDF)
        - 'euclidean': Euclidean distance
        - 'manhattan': Manhattan distance
        Can also be a callable that takes two arrays and returns a distance.
        
    random_state : int, RandomState instance or None, optional (default=42)
        Random seed for reproducibility. UMAP uses stochastic optimization,
        so results will vary between runs unless random_state is fixed.
        Set to None for non-deterministic behavior.
        
    verbose : bool, optional (default=False)
        If True, print progress messages during fitting.
        
    Returns:
    --------
    embeddings : ndarray of shape (n_samples, n_components)
        The UMAP embeddings of the input data. Each row represents the
        embedded coordinates for one input sample.
        
    umap_model : UMAP
        The fitted UMAP model object. Can be used to:
        - Transform new data: umap_model.transform(new_X)
        - Access the nearest neighbor graph: umap_model.graph_
        - Inverse transform: umap_model.inverse_transform(embeddings)
        
    Raises:
    -------
    ValueError
        - If X is None or empty
        - If n_neighbors >= n_samples (not enough data points)
        - If n_components is invalid
        
    RuntimeError
        If UMAP fitting fails for any reason
        
    """
    # Input validation
    if X is None:
        raise ValueError("Input X cannot be None")
    
    # Check if X has shape attribute (numpy array or sparse matrix)
    if not hasattr(X, 'shape'):
        raise ValueError("Input X must be a numpy array or sparse matrix")
    
    n_samples, n_features = X.shape
    
    if n_samples == 0:
        raise ValueError("Input X cannot be empty (0 samples)")
    
    if n_features == 0:
        raise ValueError("Input X cannot have 0 features")
    
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be less than the number of "
            f"samples ({n_samples}). Try reducing n_neighbors or providing "
            f"more training samples."
        )
    
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    
    if n_components > n_features:
        raise ValueError(
            f"n_components ({n_components}) cannot be greater than the "
            f"number of features ({n_features})"
        )
    
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError(f"min_dist must be between 0.0 and 1.0, got {min_dist}")
    
    # Fit UMAP model
    try:
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=verbose
        )
        
        embeddings = umap_model.fit_transform(X)
        
        return embeddings, umap_model
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate UMAP embeddings: {str(e)}")

def get_tsne_embeddings(X, n_components=2, perplexity=None, early_exaggeration=12, 
                        learning_rate='auto', n_iter=1000, metric='euclidean',
                        random_state=None, verbose=0):
    """
    Generate t-SNE embeddings for dimensionality reduction and visualization of text data.
    
    This function applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to
    reduce high-dimensional text features into a lower-dimensional space, primarily
    for visualization purposes. t-SNE is particularly effective at preserving local
    structure and revealing clusters in the data.
    
    Important: t-SNE is computationally expensive and does not support transforming
    new data points. For large datasets (>10,000 samples), consider using UMAP instead.
    
    Parameters:
    -----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        Input features to embed. Typically a TF-IDF matrix, count matrix, or
        other numerical representation of text. Sparse matrices will be converted
        to dense arrays (may require significant memory for large matrices).
        
    n_components : int, optional (default=2)
        Dimension of the output embedding space. Typical values:
        - 2: For 2D visualization (most common)
        - 3: For 3D visualization
        Values > 3 are possible but defeat t-SNE's purpose as a visualization tool.
        
    perplexity : float or None, optional (default=None)
        Related to the number of nearest neighbors used in the algorithm. This
        parameter balances local vs. global aspects of the data. If None, defaults
        to min(30, n_samples - 1).
        - Recommended range: 5 to 50
        - Smaller values: Focus on local structure
        - Larger values: Focus on global structure
        Must be less than the number of samples.
        
    early_exaggeration : float, optional (default=12)
        Controls how tight natural clusters are in the embedded space and how much
        space there is between them. Higher values create more space between clusters.
        Typical range: 8 to 20.
        
    learning_rate : float or 'auto', optional (default='auto')
        Learning rate for the optimization. If 'auto', uses max(200, n_samples / 12).
        - Typical range: 10 to 1000
        - Too high: Poor convergence or unstable results
        - Too low: Slow convergence, may get stuck in local minima
        
    n_iter : int, optional (default=1000)
        Maximum number of iterations for optimization. Increase if the algorithm
        hasn't converged (check via kl_divergence_ attribute of returned model).
        Typical range: 1000 to 5000 for good convergence.
        
    metric : str or callable, optional (default='euclidean')
        Distance metric to use for computing distances in the input space.
        Common options:
        - 'euclidean': Standard Euclidean distance
        - 'cosine': Cosine distance (can be better for text data)
        - 'manhattan': Manhattan distance
        Can also be a callable that takes two arrays and returns a distance.
        
    random_state : int, RandomState instance or None, optional (default=None)
        Random seed for reproducibility. t-SNE results can vary significantly
        between runs due to random initialization. Set to an integer (e.g., 42)
        for reproducible results, or leave as None for non-deterministic behavior.
        
    verbose : int, optional (default=0)
        Verbosity level:
        - 0: Silent (no output)
        - 1: Progress updates every 50 iterations
        - 2: Detailed progress information
        
    Returns:
    --------
    embeddings : ndarray of shape (n_samples, n_components)
        The t-SNE embeddings of the input data. Each row represents the
        embedded coordinates for one input sample.
        
    tsne_model : TSNE
        The fitted t-SNE model object. Useful for accessing:
        - Final KL divergence: tsne_model.kl_divergence_
        - Number of iterations run: tsne_model.n_iter_
        Note: Unlike UMAP, t-SNE does not support transforming new data.
        
    Raises:
    -------
    ValueError
        - If X is None, empty, or has invalid dimensions
        - If perplexity >= n_samples (not enough data points)
        - If perplexity is too small (< 5)
        - If n_components is invalid
        
    RuntimeError
        If t-SNE fitting fails for any reason
    """
    # Input validation
    if X is None:
        raise ValueError("Input X cannot be None")
    
    if not hasattr(X, 'shape'):
        raise ValueError("Input X must be a numpy array or sparse matrix")
    
    n_samples, n_features = X.shape
    
    if n_samples == 0:
        raise ValueError("Input X cannot be empty (0 samples)")
    
    if n_features == 0:
        raise ValueError("Input X cannot have 0 features")
    
    # Set default perplexity if not provided
    if perplexity is None:
        perplexity = min(30, n_samples - 1)
        if verbose > 0:
            print(f"Using default perplexity: {perplexity}")
    
    # Validate perplexity
    if perplexity >= n_samples:
        raise ValueError(
            f"perplexity ({perplexity}) must be less than the number of "
            f"samples ({n_samples}). Try reducing perplexity to at most "
            f"{n_samples - 1}, or provide more training samples."
        )
    
    if perplexity < 5:
        raise ValueError(
            f"perplexity ({perplexity}) is too small. Recommended minimum is 5. "
            f"Small perplexity values may lead to poor embeddings."
        )
    
    # Validate n_components
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    
    if n_components > 3:
        import warnings
        warnings.warn(
            f"n_components={n_components} is unusual for t-SNE. "
            f"t-SNE is primarily intended for visualization with 2 or 3 dimensions.",
            UserWarning
        )
    
    # Validate other parameters
    if early_exaggeration < 1:
        raise ValueError(f"early_exaggeration must be >= 1, got {early_exaggeration}")
    
    if n_iter < 250:
        raise ValueError(
            f"n_iter ({n_iter}) is too small. Recommended minimum is 250. "
            f"Consider using at least 1000 for good convergence."
        )
    
    # Convert sparse to dense if needed (with memory warning for large matrices)
    if hasattr(X, 'toarray'):
        memory_gb = (n_samples * n_features * 8) / (1024**3)  # 8 bytes per float64
        if memory_gb > 1 and verbose >= 0:
            print(f"Warning: Converting sparse matrix to dense (~{memory_gb:.2f} GB memory required)")
        X = X.toarray()
    
    # Fit t-SNE model
    try:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=random_state,
            verbose=verbose
        )
        
        embeddings = tsne.fit_transform(X)
        
        # Optionally report convergence info
        if verbose > 0:
            print(f"t-SNE completed in {tsne.n_iter_} iterations")
            print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
        
        return embeddings, tsne
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate t-SNE embeddings: {str(e)}")
def get_sbert_embeddings(text, model=None, model_name='all-MiniLM-L6-v2', 
                         batch_size=32, show_progress_bar=False, 
                         normalize_embeddings=False, device=None):
    """
    Generate Sentence-BERT (SBERT) embeddings for text documents.
    
    This function uses pretrained Sentence-BERT models to generate dense vector
    embeddings that capture semantic meaning of text. SBERT embeddings are
    particularly effective for:
    - Semantic search and similarity comparisons
    - Clustering semantically similar documents
    - Document classification tasks
    
    The function supports model reuse to avoid expensive reloading on repeated calls.
    
    Parameters:
    -----------
    text : array-like of str
        Collection of text documents to encode. Each element should be a string
        representing a sentence, paragraph, or document. Can be a list, numpy
        array, pandas Series, or any iterable containing text strings.
        
    model : SentenceTransformer or None, optional (default=None)
        Pre-loaded SentenceTransformer model instance. If provided, this model
        is used directly, avoiding the overhead of loading a model from disk.
        If None, a new model is loaded using the model_name parameter.
        
        For efficiency when calling this function multiple times, load the model
        once and pass it to subsequent calls:
        ```
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1, _ = get_sbert_embeddings(texts1, model=model)
        emb2, _ = get_sbert_embeddings(texts2, model=model)
        ```
        
    model_name : str, optional (default='all-MiniLM-L6-v2')
        Name of the pretrained Sentence-BERT model to load. Only used if model
        parameter is None. Popular options:
        - 'all-MiniLM-L6-v2': Fast, good quality, 384 dimensions (recommended)
        - 'all-mpnet-base-v2': Higher quality, slower, 768 dimensions
        - 'paraphrase-MiniLM-L6-v2': Optimized for paraphrase detection
        - 'multi-qa-MiniLM-L6-cos-v1': Optimized for question-answering
        
        See https://www.sbert.net/docs/pretrained_models.html for full list.
        
    batch_size : int, optional (default=32)
        Number of texts to encode in parallel. Larger batch sizes are faster
        but require more memory:
        - GPU: Can typically handle 32-128 depending on text length and VRAM
        - CPU: Usually 8-32 is reasonable
        Reduce if encountering out-of-memory errors.
        
    show_progress_bar : bool, optional (default=False)
        If True, displays a progress bar during encoding. Useful for large
        datasets to monitor progress.
        
    normalize_embeddings : bool, optional (default=False)
        If True, normalizes embeddings to unit length (L2 norm = 1). This is
        recommended when using cosine similarity, as it allows using faster
        dot product instead of full cosine similarity computation.
        
    device : str or None, optional (default=None)
        Device to use for computation:
        - 'cuda' or 'cuda:0': Use GPU (much faster if available)
        - 'cpu': Use CPU only
        - None: Automatically detect and use GPU if available, otherwise CPU
        
    Returns:
    --------
    embeddings : ndarray of shape (n_texts, embedding_dim)
        Dense vector embeddings for each input text. The embedding dimension
        depends on the model:
        - all-MiniLM-L6-v2: 384 dimensions
        - all-mpnet-base-v2: 768 dimensions
        Each row represents the embedding for one input text.
        
    model : SentenceTransformer
        The SentenceTransformer model used for encoding. Return this and reuse
        it in subsequent calls to avoid reloading the model:
        ```
        embeddings1, model = get_sbert_embeddings(texts1)
        embeddings2, _ = get_sbert_embeddings(texts2, model=model)
        ```
        
    Raises:
    -------
    ValueError
        - If text is None, empty, or contains non-string elements
        - If text is not iterable
        
    RuntimeError
        - If model loading fails (e.g., invalid model_name, network issues)
        - If encoding fails (e.g., out of memory, incompatible inputs)
        
    Examples:
    ---------
    >>> # Basic usage - model loads once
    >>> texts = [
    ...     "The quick brown fox jumps over the lazy dog",
    ...     "A fast auburn fox leaps above a sleepy canine",
    ...     "Python is a programming language"
    ... ]
    >>> embeddings, model = get_sbert_embeddings(texts)
    >>> print(f"Embedding shape: {embeddings.shape}")
    >>> print(f"Embedding dimension: {embeddings.shape[1]}")
    
    >>> # Efficient repeated usage - reuse loaded model
    >>> more_texts = ["Machine learning is fascinating", "AI advances rapidly"]
    >>> more_embeddings, _ = get_sbert_embeddings(more_texts, model=model)
    
    >>> # Use a different model with GPU and normalization
    >>> embeddings, model = get_sbert_embeddings(
    ...     texts,
    ...     model_name='all-mpnet-base-v2',
    ...     device='cuda',
    ...     normalize_embeddings=True,
    ...     show_progress_bar=True
    ... )
    
    >>> # Compute semantic similarity between texts
    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> embeddings, model = get_sbert_embeddings(texts, normalize_embeddings=True)
    >>> similarities = cosine_similarity(embeddings)
    >>> print(f"Similarity between text 0 and 1: {similarities[0, 1]:.3f}")
    
    >>> # Large dataset with progress monitoring
    >>> large_corpus = ["document " + str(i) for i in range(10000)]
    >>> embeddings, model = get_sbert_embeddings(
    ...     large_corpus,
    ...     batch_size=64,
    ...     show_progress_bar=True
    ... )
    
    Notes:
    ------
    - **Performance**: Loading a model takes 1-5 seconds and ~90-300MB memory.
      Always reuse the model when calling this function multiple times.
    - **GPU Acceleration**: Using a GPU can be 10-50x faster than CPU. Specify
      device='cuda' to use GPU if available.
    - **Memory**: Each text embedding requires 384-768 floats (1.5-3 KB) depending
      on the model. A corpus of 100,000 texts needs ~150-300 MB for embeddings.
    - **Text Length**: Most models have a maximum token limit (usually 128-512 tokens).
      Longer texts are truncated. Consider splitting very long documents.
    - **Normalization**: Enable normalize_embeddings=True when using cosine
      similarity for semantic search, as it enables faster dot product computation.
    - **Model Selection**: For most tasks, 'all-MiniLM-L6-v2' offers the best
      speed/quality tradeoff. Use 'all-mpnet-base-v2' for higher quality at the
      cost of 2-3x slower encoding.
    - **Batch Size**: Larger batch sizes are faster but use more memory. If you
      get CUDA out-of-memory errors, reduce batch_size.
    """
    # Input validation
    if text is None:
        raise ValueError("Input text cannot be None")
    
    try:
        text_list = list(text)
    except (TypeError, ValueError):
        raise ValueError("Text must be iterable (e.g., list, array, pandas Series)")
    
    if not text_list:
        raise ValueError("Text cannot be empty - must contain at least one document")
    
    if not all(isinstance(doc, str) for doc in text_list):
        raise ValueError(
            "All elements in text must be strings. "
            f"Found types: {set(type(doc).__name__ for doc in text_list)}"
        )
    
    # Validate batch_size
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    
    # Load model if not provided
    model_was_loaded = False
    if model is None:
        try:
            if device is None:
                # Auto-detect GPU
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            model = SentenceTransformer(model_name, device=device)
            model_was_loaded = True
            
        except ImportError:
            # torch not available, use CPU
            try:
                model = SentenceTransformer(model_name, device='cpu')
                model_was_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{model_name}': {str(e)}\n"
                    f"Make sure the model name is correct and you have internet connectivity "
                    f"for first-time downloads."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{model_name}': {str(e)}\n"
                f"Common issues:\n"
                f"  - Invalid model name (see https://www.sbert.net/docs/pretrained_models.html)\n"
                f"  - Network issues during first-time model download\n"
                f"  - Insufficient disk space for model cache"
            )
    
    # Encode text
    try:
        embeddings = model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings, model
        
    except Exception as e:
        error_msg = f"Failed to encode text with SBERT: {str(e)}"
        
        # Provide helpful error messages for common issues
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            error_msg += (
                f"\n\nOut of memory error. Try:\n"
                f"  1. Reducing batch_size (current: {batch_size})\n"
                f"  2. Using a smaller model (e.g., 'all-MiniLM-L6-v2')\n"
                f"  3. Processing texts in smaller chunks"
            )
        elif "cuda" in str(e).lower() and device == 'cuda':
            error_msg += (
                f"\n\nGPU error detected. Try:\n"
                f"  1. Setting device='cpu' to use CPU instead\n"
                f"  2. Checking if CUDA is properly installed"
            )
        
        raise RuntimeError(error_msg)

def get_word2vec_embeddings(text, vector_size=100, window=5, min_count=1, 
                            sg=0, workers=4, epochs=5, seed=None):
    """
    Generate Word2Vec embeddings for tokenized text documents.
    
    This function trains a Word2Vec model on tokenized text and creates document-level
    embeddings by averaging the word vectors for each document. Word2Vec learns
    distributed representations of words based on their context, capturing semantic
    relationships between words.
    
    The function handles documents with out-of-vocabulary words gracefully and
    provides warnings when many documents cannot be properly embedded.
    
    Parameters:
    -----------
    text : list of list of str
        Collection of tokenized documents. Each document should be a list of
        word tokens (strings). Example format:
        [
            ['this', 'is', 'first', 'document'],
            ['this', 'is', 'second', 'document'],
            ['another', 'example', 'text']
        ]
        
        The text must be pre-tokenized (split into words). Use tokenization
        methods like:
        - text.lower().split() for simple splitting
        - nltk.word_tokenize() for better tokenization
        - Custom tokenizers for domain-specific needs
        
    vector_size : int, optional (default=100)
        Dimensionality of the word vectors and resulting document embeddings.
        Common values:
        - 50-100: Faster training, less memory, good for small datasets
        - 200-300: Standard choice, good balance
        - 300-500: Higher quality for large datasets
        Must be >= 1.
        
    window : int, optional (default=5)
        Maximum distance between the current and predicted word within a sentence.
        Larger windows capture more global context, smaller windows focus on
        local syntax.
        - Small (2-3): Focuses on syntactic relationships
        - Medium (5-7): Balanced (recommended)
        - Large (10-15): Captures broader semantic relationships
        Must be >= 1.
        
    min_count : int, optional (default=1)
        Ignores all words with total frequency lower than this. Higher values
        reduce vocabulary size and training time, but may exclude rare but
        meaningful words.
        - 1: Keep all words (good for small datasets)
        - 5-10: Filter rare words (good for large datasets)
        Must be >= 1.
        
    sg : int, optional (default=0)
        Training algorithm:
        - 0: CBOW (Continuous Bag of Words) - faster, better for frequent words
        - 1: Skip-gram - slower, better for rare words and small datasets
        
        Generally, use CBOW for large datasets and Skip-gram for small datasets
        or when you care about rare words.
        
    workers : int, optional (default=4)
        Number of CPU threads to use for training. Higher values speed up
        training on multi-core machines. Set to 1 for deterministic training
        (even with seed set, multi-threading can cause slight variations).
        
    epochs : int, optional (default=5)
        Number of iterations (epochs) over the corpus. More epochs generally
        improve quality but increase training time.
        - 3-5: Quick training for experimentation
        - 10-20: Better quality for production use
        - 50+: Maximum quality for important applications
        
    seed : int or None, optional (default=None)
        Random seed for reproducibility. Word2Vec training is stochastic, so
        results vary between runs. Set to an integer (e.g., 42) for reproducible
        results, or leave as None for non-deterministic training.
        
        Note: Even with a seed, using workers > 1 may cause slight variations
        due to multi-threading non-determinism.
        
    Returns:
    --------
    embeddings : ndarray of shape (n_documents, vector_size)
        Document embeddings created by averaging word vectors for each document.
        Each row represents one document's embedding. Documents with no words
        in the vocabulary receive zero vectors.
        
    model : Word2Vec
        The trained Word2Vec model. Can be used to:
        - Get word vectors: model.wv['word']
        - Find similar words: model.wv.most_similar('word')
        - Check vocabulary: model.wv.key_to_index
        - Encode new documents using the same vocabulary
        
    Raises:
    -------
    ValueError
        - If text is None, empty, or not properly formatted
        - If text is not a list of lists of strings
        - If any parameter has an invalid value
        
    RuntimeError
        If Word2Vec training fails
        
    Warnings:
    ---------
    UserWarning
        If more than 10% of documents have no words in vocabulary (receive
        zero vectors), suggesting potential issues with tokenization or
        min_count setting.
        
    Examples:
    ---------
    >>> # Basic usage with pre-tokenized text
    >>> documents = [
    ...     ['machine', 'learning', 'is', 'fascinating'],
    ...     ['deep', 'learning', 'uses', 'neural', 'networks'],
    ...     ['natural', 'language', 'processing', 'is', 'challenging']
    ... ]
    >>> embeddings, model = get_word2vec_embeddings(documents)
    >>> print(f"Document embeddings shape: {embeddings.shape}")
    >>> print(f"Vocabulary size: {len(model.wv)}")
    
    >>> # Access individual word vectors
    >>> if 'learning' in model.wv:
    ...     word_vector = model.wv['learning']
    ...     print(f"Vector for 'learning': {word_vector[:5]}")
    
    >>> # Find similar words
    >>> similar_words = model.wv.most_similar('learning', topn=3)
    >>> print(f"Words similar to 'learning': {similar_words}")
    
    >>> # Use Skip-gram with higher dimensions and reproducibility
    >>> embeddings, model = get_word2vec_embeddings(
    ...     documents,
    ...     vector_size=200,
    ...     window=7,
    ...     min_count=1,
    ...     sg=1,  # Skip-gram
    ...     epochs=10,
    ...     seed=42
    ... )
    
    >>> # Encode new documents using the trained model
    >>> def encode_new_document(doc_tokens, model):
    ...     vectors = [model.wv[word] for word in doc_tokens if word in model.wv]
    ...     if vectors:
    ...         return np.mean(vectors, axis=0)
    ...     else:
    ...         return np.zeros(model.vector_size)
    ...
    >>> new_doc = ['artificial', 'intelligence', 'applications']
    >>> new_embedding = encode_new_document(new_doc, model)
    
    >>> # Real-world example with text preprocessing
    >>> raw_texts = [
    ...     "Machine learning is a subset of AI.",
    ...     "Deep learning requires large datasets.",
    ...     "NLP focuses on human language understanding."
    ... ]
    >>> # Tokenize: lowercase and split
    >>> tokenized = [text.lower().split() for text in raw_texts]
    >>> embeddings, model = get_word2vec_embeddings(
    ...     tokenized,
    ...     vector_size=150,
    ...     window=5,
    ...     min_count=1,
    ...     sg=0,
    ...     workers=2,
    ...     epochs=10,
    ...     seed=42
    ... )
    
    Notes:
    ------
    - **Tokenization is required**: Text must be pre-tokenized into lists of words.
      The function does not perform tokenization automatically.
    
    - **Document Embeddings**: This function creates document embeddings by
      averaging word vectors. More sophisticated methods exist (Doc2Vec, weighted
      averaging by TF-IDF, etc.) but simple averaging often works well.
    
    - **Vocabulary**: Words appearing fewer than min_count times are excluded
      from the vocabulary. This affects which words contribute to document
      embeddings.
    
    - **Zero Vectors**: Documents containing only out-of-vocabulary words (or
      empty documents) receive zero vectors, which may affect downstream analysis.
    
    - **Training Time**: Word2Vec training is relatively fast. A corpus of
      10,000 documents with 100-word average length trains in seconds on a
      modern CPU.
    
    - **CBOW vs Skip-gram**:
      * CBOW (sg=0): Predicts target word from context. Faster, better for
        frequent words, generally preferred for large datasets.
      * Skip-gram (sg=1): Predicts context from target word. Better for rare
        words and smaller datasets, but slower to train.
    
    - **Reproducibility**: For fully deterministic results, set both seed and
      workers=1. Using workers > 1 may introduce slight variations even with
      a fixed seed due to thread scheduling non-determinism.
    
    - **Memory**: Word2Vec memory usage is proportional to vocabulary size ×
      vector_size. A vocabulary of 10,000 words with vector_size=100 uses
      ~4 MB for word vectors.
    
    - **Comparison with other methods**:
      * Word2Vec: Fast, interpretable, good for smaller datasets
      * FastText: Better for morphologically rich languages, handles OOV words
      * BERT/SBERT: Higher quality but much slower and more complex
    """
    # Input validation
    if text is None:
        raise ValueError("Input text cannot be None")
    
    try:
        text_list = list(text)
    except (TypeError, ValueError):
        raise ValueError("Text must be iterable")
    
    if not text_list:
        raise ValueError(
            "Text cannot be empty. Provide at least one tokenized document."
        )
    
    # Validate tokenization format (list of lists)
    if not all(isinstance(doc, (list, tuple)) for doc in text_list):
        raise ValueError(
            "Text must be a list of tokenized documents (list of lists of strings). "
            "Each document should be a list/tuple of word tokens. "
            "Example: [['word1', 'word2'], ['word3', 'word4']]\n"
            "Did you forget to tokenize your text?"
        )
    
    # Check if any documents are empty
    empty_docs = sum(1 for doc in text_list if len(doc) == 0)
    if empty_docs > 0:
        import warnings
        warnings.warn(
            f"{empty_docs}/{len(text_list)} documents are empty. "
            f"These will receive zero vectors.",
            UserWarning
        )
    
    # Validate all tokens are strings
    non_string_found = False
    for doc_idx, doc in enumerate(text_list):
        for token in doc:
            if not isinstance(token, str):
                non_string_found = True
                raise ValueError(
                    f"All tokens must be strings. Found {type(token).__name__} "
                    f"in document {doc_idx}: {token}"
                )
    
    # Validate parameters
    if vector_size < 1:
        raise ValueError(f"vector_size must be >= 1, got {vector_size}")
    
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    
    if sg not in (0, 1):
        raise ValueError(f"sg must be 0 (CBOW) or 1 (Skip-gram), got {sg}")
    
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    
    # Count total words to warn about small datasets
    total_words = sum(len(doc) for doc in text_list)
    if total_words < 100:
        import warnings
        warnings.warn(
            f"Very small corpus ({total_words} total words). "
            f"Word2Vec works best with at least 1000+ words. "
            f"Results may be poor quality.",
            UserWarning
        )
    
    # Train Word2Vec model
    try:
        model = Word2Vec(
            sentences=text_list,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            epochs=epochs,
            seed=seed
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to train Word2Vec model: {str(e)}\n"
            f"Check that your text is properly tokenized and contains enough data."
        )
    
    # Check vocabulary size
    vocab_size = len(model.wv)
    if vocab_size == 0:
        raise RuntimeError(
            "Word2Vec produced empty vocabulary. This usually means:\n"
            f"  1. min_count ({min_count}) is too high for your corpus size\n"
            f"  2. Text is not properly tokenized\n"
            f"  3. All documents are empty"
        )
    
    # Create document embeddings efficiently
    embeddings = []
    docs_with_no_vectors = 0
    
    for doc in text_list:
        if len(doc) == 0:
            # Empty document
            embeddings.append(np.zeros(vector_size))
            docs_with_no_vectors += 1
            continue
        
        # Efficiently collect vectors for words in vocabulary
        doc_vectors = []
        for word in doc:
            try:
                doc_vectors.append(model.wv[word])
            except KeyError:
                # Word not in vocabulary (filtered by min_count or not seen)
                pass
        
        if doc_vectors:
            # Average word vectors to create document embedding
            doc_embedding = np.mean(doc_vectors, axis=0)
        else:
            # No words in vocabulary - use zero vector
            doc_embedding = np.zeros(vector_size)
            docs_with_no_vectors += 1
        
        embeddings.append(doc_embedding)
    
    # Warning if many documents have no vectors
    if docs_with_no_vectors > len(text_list) * 0.1:  # More than 10%
        import warnings
        warnings.warn(
            f"{docs_with_no_vectors}/{len(text_list)} documents ({docs_with_no_vectors/len(text_list)*100:.1f}%) "
            f"had no words in vocabulary and received zero vectors. Consider:\n"
            f"  1. Reducing min_count (current: {min_count})\n"
            f"  2. Checking if text is properly tokenized\n"
            f"  3. Using more/better training data\n"
            f"  4. Checking for empty documents\n"
            f"Vocabulary size: {vocab_size} words",
            UserWarning
        )
    
    return np.array(embeddings), model
def get_pretrained_word_embeddings(text, model=None, model_name='glove-wiki-gigaword-300'):
    """
    Generate document embeddings using pretrained word vectors (GloVe, Word2Vec, FastText).
    
    This function uses pretrained word embeddings from the Gensim-data repository to
    create document-level embeddings by averaging word vectors. Pretrained embeddings
    are trained on massive corpora and capture rich semantic relationships without
    requiring training on your specific dataset.
    
    IMPORTANT: The first time a model is loaded, it will be downloaded from the
    internet (100MB-1.6GB depending on model) and cached locally. Subsequent loads
    are much faster. Always reuse loaded models when processing multiple batches.
    
    Parameters:
    -----------
    text : list of list of str
        Collection of tokenized documents. Each document should be a list of
        word tokens (strings). Example format:
        [
            ['this', 'is', 'first', 'document'],
            ['this', 'is', 'second', 'document'],
            ['another', 'example', 'text']
        ]
        
        The text must be pre-tokenized (split into words). Pretrained models
        are typically trained on lowercase text, so consider lowercasing your
        tokens for better vocabulary coverage.
        
    model : KeyedVectors or None, optional (default=None)
        Pre-loaded word vector model from Gensim. If provided, this model is
        used directly, avoiding the overhead of loading from disk/network.
        If None, a new model is loaded using the model_name parameter.
        
        CRITICAL FOR EFFICIENCY: When calling this function multiple times,
        load the model once and pass it to subsequent calls:
        ```
        embeddings1, model = get_pretrained_word_embeddings(docs1)
        embeddings2, _ = get_pretrained_word_embeddings(docs2, model=model)
        embeddings3, _ = get_pretrained_word_embeddings(docs3, model=model)
        ```
        This avoids reloading a 100MB-1.6GB model every time.
        
    model_name : str, optional (default='glove-wiki-gigaword-300')
        Name of the pretrained model to load from Gensim-data. Only used if
        the model parameter is None. Available models include:
        
        **GloVe (Global Vectors for Word Representation):**
        - 'glove-wiki-gigaword-50': 50-dim, 6B tokens, 400K vocab, ~65MB
        - 'glove-wiki-gigaword-100': 100-dim, 6B tokens, 400K vocab, ~128MB
        - 'glove-wiki-gigaword-200': 200-dim, 6B tokens, 400K vocab, ~252MB
        - 'glove-wiki-gigaword-300': 300-dim, 6B tokens, 400K vocab, ~376MB (default)
        - 'glove-twitter-25': 25-dim, 2B tweets, 1.2M vocab, ~100MB
        - 'glove-twitter-50': 50-dim, 2B tweets, 1.2M vocab, ~200MB
        - 'glove-twitter-100': 100-dim, 2B tweets, 1.2M vocab, ~400MB
        - 'glove-twitter-200': 200-dim, 2B tweets, 1.2M vocab, ~800MB
        
        **Word2Vec:**
        - 'word2vec-google-news-300': 300-dim, 3M vocab, ~1.6GB
          (trained on Google News, very large vocabulary)
        
        **FastText:**
        - 'fasttext-wiki-news-subwords-300': 300-dim, 999MB
          (includes subword information, handles out-of-vocabulary words better)
        
        **Conceptnet Numberbatch:**
        - 'conceptnet-numberbatch-17-06-300': 300-dim, multilingual, ~1.1GB
        
        For a complete list with descriptions, run:
        ```python
        import gensim.downloader as api
        print(api.info())
        ```
        
    Returns:
    --------
    embeddings : ndarray of shape (n_documents, vector_size)
        Document embeddings created by averaging word vectors for each document.
        Each row represents one document's embedding. Documents with no words
        in the vocabulary receive zero vectors. The vector_size depends on the
        model (e.g., 50, 100, 200, or 300).
        
    model : KeyedVectors
        The loaded word vector model. Can be used to:
        - Get word vectors: model['word']
        - Find similar words: model.most_similar('word')
        - Check vocabulary: 'word' in model
        - Get vocabulary size: len(model)
        - Reuse for encoding more documents
        
        IMPORTANT: Save and reuse this model to avoid reloading on every call.
        
    Raises:
    -------
    ValueError
        - If text is None, empty, or not properly formatted
        - If text is not a list of lists of strings
        
    RuntimeError
        - If model loading fails (invalid model_name, network issues, etc.)
        
    Warnings:
    ---------
    UserWarning
        If more than 10% of documents have no words in vocabulary (receive
        zero vectors), suggesting potential tokenization or vocabulary issues.
        
    Notes:
    ------
    - **Model Selection Guidelines**:
      * **General text**: glove-wiki-gigaword-100 or glove-wiki-gigaword-300
      * **Social media/informal**: glove-twitter-* models
      * **Large vocabulary needed**: word2vec-google-news-300
      * **Out-of-vocabulary handling**: fasttext-wiki-news-subwords-300
      * **Speed critical**: glove-wiki-gigaword-50 or glove-twitter-25
      * **Quality critical**: glove-wiki-gigaword-300 or word2vec-google-news-300
    
    - **Comparison with Training Your Own**:
      * Pretrained: No training needed, immediate use, good general knowledge
      * Custom Word2Vec: Requires training data, domain-specific, can capture
        unique vocabulary and relationships in your specific domain
    
    - **FastText Advantage**: Unlike GloVe/Word2Vec, FastText includes subword
      information, so it can generate vectors for out-of-vocabulary words by
      composing character n-grams. Use 'fasttext-wiki-news-subwords-300' if
      you have many rare or misspelled words.
    """
  
    
    # Input validation
    if text is None:
        raise ValueError("Input text cannot be None")
    
    try:
        text_list = list(text)
    except (TypeError, ValueError):
        raise ValueError("Text must be iterable")
    
    if not text_list:
        raise ValueError(
            "Text cannot be empty. Provide at least one tokenized document."
        )
    
    # Validate tokenization format (list of lists)
    if not all(isinstance(doc, (list, tuple)) for doc in text_list):
        raise ValueError(
            "Text must be a list of tokenized documents (list of lists of strings). "
            "Each document should be a list/tuple of word tokens. "
            "Example: [['word1', 'word2'], ['word3', 'word4']]\n"
            "Did you forget to tokenize your text?"
        )
    
    # Check if any documents are empty
    empty_docs = sum(1 for doc in text_list if len(doc) == 0)
    if empty_docs > 0:
        import warnings
        warnings.warn(
            f"{empty_docs}/{len(text_list)} documents are empty. "
            f"These will receive zero vectors.",
            UserWarning
        )
    
    # Validate all tokens are strings
    for doc_idx, doc in enumerate(text_list):
        for token in doc:
            if not isinstance(token, str):
                raise ValueError(
                    f"All tokens must be strings. Found {type(token).__name__} "
                    f"in document {doc_idx}: {token}"
                )
    
    # Load model if not provided
    model_was_loaded = False
    if model is None:
        try:
            print(f"Loading pretrained model '{model_name}'...")
            print("(This may take a while on first run - downloading and caching model)")
            model = api.load(model_name)
            model_was_loaded = True
            print(f"  Model loaded successfully!")
            print(f"  Vocabulary size: {len(model):,} words")
            print(f"  Vector dimension: {model.vector_size}")
            
        except Exception as e:
            error_msg = f"Failed to load pretrained model '{model_name}': {str(e)}\n\n"
            
           
            if "not found" in str(e).lower() or "unknown" in str(e).lower():
                error_msg += (
                    "Model name not recognized. Available models include:\n"
                    "  - 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', "
                    "'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300'\n"
                    "  - 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', "
                    "'glove-twitter-200'\n"
                    "  - 'word2vec-google-news-300'\n"
                    "  - 'fasttext-wiki-news-subwords-300'\n"
                    "  - 'conceptnet-numberbatch-17-06-300'\n\n"
                    "For the complete list, run:\n"
                    "  import gensim.downloader as api\n"
                    "  print(api.info())"
                )
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                error_msg += (
                    "Network error - check your internet connection.\n"
                    "Models are downloaded from the internet on first use."
                )
            
            raise RuntimeError(error_msg)
    
    # Get vector dimension from loaded model
    vector_size = model.vector_size
    
    # Create document embeddings efficiently
    embeddings = []
    docs_with_no_vectors = 0
    total_words = 0
    words_in_vocab = 0
    
    for doc in text_list:
        if len(doc) == 0:
            # Empty document
            embeddings.append(np.zeros(vector_size))
            docs_with_no_vectors += 1
            continue
        
        # Efficiently collect vectors for words in vocabulary
        doc_vectors = []
        for word in doc:
            total_words += 1
            try:
                doc_vectors.append(model[word])
                words_in_vocab += 1
            except KeyError:
                # Word not in vocabulary
                pass
        
        if doc_vectors:
            # Average word vectors to create document embedding
            doc_embedding = np.mean(doc_vectors, axis=0)
        else:
            # No words in vocabulary - use zero vector
            doc_embedding = np.zeros(vector_size)
            docs_with_no_vectors += 1
        
        embeddings.append(doc_embedding)
    
    # Calculate vocabulary coverage
    vocab_coverage = words_in_vocab / total_words if total_words > 0 else 0
    
    # Warning if many documents have no vectors
    if docs_with_no_vectors > len(text_list) * 0.1:  # More than 10%
        import warnings
        warnings.warn(
            f"{docs_with_no_vectors}/{len(text_list)} documents ({docs_with_no_vectors/len(text_list)*100:.1f}%) "
            f"had no words in vocabulary and received zero vectors.\n"
            f"Vocabulary coverage: {vocab_coverage:.1%} of words found in model.\n"
            f"Consider:\n"
            f"  1. Lowercasing your text (most pretrained models use lowercase)\n"
            f"  2. Using a model with larger vocabulary (e.g., 'word2vec-google-news-300')\n"
            f"  3. Using FastText for better out-of-vocabulary handling\n"
            f"  4. Checking if text is properly tokenized",
            UserWarning
        )
    elif model_was_loaded:
        # Show coverage info when model was just loaded
        print(f"  Vocabulary coverage: {vocab_coverage:.1%} of input words found in model")
    
    return np.array(embeddings), model

def get_lsa_embeddings(X, n_components=100, random_state=None, algorithm='randomized', 
                       n_iter=5, tol=0.0):
    """
    Generate Latent Semantic Analysis (LSA) embeddings using Truncated SVD.
    
    LSA (also known as Latent Semantic Indexing) uses Singular Value Decomposition
    to reduce the dimensionality of document-term matrices while capturing the most
    important latent semantic structures.
    
    Parameters:
    -----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        Input document-term matrix.
        
    n_components : int, optional (default=100)
        Number of dimensions for the output embedding space. This corresponds
        to the number of latent topics/concepts to extract.
        
    random_state : int, RandomState instance or None, optional (default=None)
        Random seed for reproducibility.
        
    algorithm : str, optional (default='randomized')
        SVD solver algorithm to use: 'randomized' or 'arpack'.
        
    n_iter : int, optional (default=5)
        Number of iterations for the randomized SVD solver. Only used when
        algorithm='randomized'.
        
    tol : float, optional (default=0.0)
        Tolerance for ARPACK solver. Only used when algorithm='arpack'.
        
    Returns:
    --------
    embeddings : ndarray of shape (n_samples, n_components_use)
        The LSA embeddings of the input documents.
        
    svd_model : TruncatedSVD
        The fitted TruncatedSVD model.
        
    Raises:
    -------
    ValueError
        If inputs are invalid or n_components is unfeasible.
        
    RuntimeError
        If SVD fitting fails for any reason.
    """
    # --- Input validation ---
    if X is None:
        raise ValueError("Input X cannot be None")
    
    if not hasattr(X, 'shape'):
        raise ValueError("Input X must be a numpy array or sparse matrix")
    
    n_samples, n_features = X.shape
    
    if n_samples == 0:
        raise ValueError("Input X cannot be empty (0 samples)")
    
    if n_features == 0:
        raise ValueError("Input X cannot have 0 features")
        
    # FIX for test_invalid_n_components: n_components must be >= 1
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    
    # --- n_components validation ---
    
    # Maximum theoretical components is min(n_samples, n_features)
    max_components_limit = min(n_samples, n_features)
    # Maximum recommended (and often required by SVD) is one less than the limit
    max_allowed_safe = max_components_limit - 1 

    # FIX for test_n_components_too_large: Raise ValueError if requested n_components 
    # exceeds the maximum possible rank of the matrix.
    if n_components >= max_components_limit:
        raise ValueError(
            f"Requested n_components={n_components} is too large. "
            f"Must be strictly less than min(n_samples, n_features) = {max_components_limit}. "
            f"Consider using n_components <= {max_allowed_safe}."
        )

    # Use the requested n_components as the starting point
    n_components_use = n_components
    
    # Check against algorithm-specific constraints (this is where adjustment or warning happens)
    if algorithm == 'arpack' and n_components_use >= n_features:
        # ARPACK technically requires k < n_features, but this is already covered by max_components_limit
        # unless n_samples > n_features. We will warn and adjust to the safe limit.
        warnings.warn(
            f"Requested n_components={n_components} is too large for features={n_features} "
            f"using algorithm='arpack', adjusting down to {max_allowed_safe}",
            UserWarning
        )
        n_components_use = max_allowed_safe
        
    # Final check just in case.
    if n_components_use < 1:
        raise ValueError(f"Adjusted n_components is less than 1. Check matrix shape: {X.shape}")
        
    # --- Validate algorithm and n_iter ---
    if algorithm not in ('randomized', 'arpack'):
        raise ValueError(
            f"algorithm must be 'randomized' or 'arpack', got '{algorithm}'"
        )
    
    if algorithm == 'randomized' and n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    
    # --- Fit TruncatedSVD model with enhanced error handling ---
    try:
        svd = TruncatedSVD(
            n_components=n_components_use,
            algorithm=algorithm,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state
        )
        
        embeddings = svd.fit_transform(X)
        
        return embeddings, svd
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Error in get_lsa_embeddings with "
            f"n_samples={n_samples}, n_features={n_features}, "
            f"requested_components={n_components}, actual_components={n_components_use}, "
            f"algorithm={algorithm}.\n"
            f"Exception: {e}\nTraceback:\n{tb}"
        )


def generate_all_embeddings(text, tokenized_text=None, models=None, 
                           embedding_params=None, verbose=True):
    """
    Generate multiple types of embeddings for text data in a single call.
    
    This convenience function generates a comprehensive set of text embeddings
    using various methods, including TF-IDF, dimensionality reduction techniques,
    neural embeddings, and word-based methods. It's designed for:
    - Rapid prototyping and experimentation with multiple embedding types
    - Comparing different embedding methods
    - Creating ensemble features for downstream tasks
    
    IMPORTANT: This function can be slow and memory-intensive as it generates
    many embeddings. For production use, generate only the embeddings you need.
    
    The function supports model reuse, which is CRITICAL for efficiency when
    working with pretrained models (SBERT, GloVe) that are expensive to load.
    
    Parameters:
    -----------
    text : list of str
        Collection of raw text documents. Each element should be a string
        representing a document. This is used for TF-IDF, dimensionality
        reduction methods (UMAP, t-SNE, LSA), and sentence-level embeddings
        (SBERT).
        
    tokenized_text : list of list of str or None, optional (default=None)
        Collection of tokenized documents for word-level embedding methods
        (Word2Vec, GloVe). Each document should be a list of word tokens.
        
        If None, word-level embeddings (word2vec, glove) will be skipped.
        
        
    models : dict or None, optional (default=None)
        Dictionary of pre-loaded models to reuse. This is CRITICAL for efficiency
        when calling this function multiple times, especially for expensive models
        like SBERT and GloVe which take 10-60 seconds to load.
        
        Keys can include:
        - 'sbert': Pre-loaded SentenceTransformer model
        - 'glove': Pre-loaded word vector model (KeyedVectors)
        
        
    embedding_params : dict or None, optional (default=None)
        Dictionary of parameters to pass to individual embedding functions.
        Allows customization of embedding generation.
        
        Keys can include:
        - 'tfidf': dict of params for vectorize_text()
        - 'umap': dict of params for get_umap_embeddings()
        - 'tsne': dict of params for get_tsne_embeddings()
        - 'lsa': dict of params for get_lsa_embeddings()
        - 'sbert': dict of params for get_sbert_embeddings()
        - 'word2vec': dict of params for get_word2vec_embeddings()
        - 'glove': dict of params for get_pretrained_word_embeddings()
        
        ```
        
    verbose : bool, optional (default=True)
        If True, prints progress messages as each embedding type is generated.
        Useful for monitoring progress on large datasets.
        
    Returns:
    --------
    embeddings : dict
        Dictionary containing all generated embeddings. Keys are embedding type
        names, values are the embedding arrays or matrices:
        
        Always included:
        - 'tfidf': TF-IDF sparse matrix (n_samples, n_features)
        - 'umap': UMAP embeddings (n_samples, n_components)
        - 'tsne': t-SNE embeddings (n_samples, n_components)
        - 'lsa': LSA embeddings (n_samples, n_components)
        - 'sbert': Sentence-BERT embeddings (n_samples, 384 or 768)
        
        Included if tokenized_text is provided:
        - 'word2vec': Word2Vec document embeddings (n_samples, vector_size)
        - 'glove': GloVe document embeddings (n_samples, vector_size)
        
    models : dict
        Dictionary containing all fitted/loaded models for reuse:
        - 'vectorizer': Fitted TfidfVectorizer
        - 'umap': Fitted UMAP model
        - 'tsne': Fitted TSNE model
        - 'lsa': Fitted TruncatedSVD model
        - 'sbert': Loaded SentenceTransformer model
        - 'word2vec': Trained Word2Vec model (if tokenized_text provided)
        - 'glove': Loaded GloVe KeyedVectors (if tokenized_text provided)
        
        CRITICAL: Save and reuse this models dict to avoid reloading expensive
        models (SBERT, GloVe) on subsequent calls.
        
    Raises:
    -------
    ValueError
        - If text is None or empty
        - If text is not a list of strings
        
    RuntimeError
        If any embedding generation fails (partial results may be returned
        depending on where the failure occurred)
        
    """
    # Input validation
    if text is None:
        raise ValueError("Input text cannot be None")
    
    try:
        text_list = list(text)
    except (TypeError, ValueError):
        raise ValueError("Text must be iterable")
    
    if not text_list:
        raise ValueError("Text cannot be empty - provide at least one document")
    
    if not all(isinstance(doc, str) for doc in text_list):
        raise ValueError("All elements in text must be strings")
    
    # Validate tokenized_text if provided
    if tokenized_text is not None:
        try:
            tokenized_list = list(tokenized_text)
        except (TypeError, ValueError):
            raise ValueError("tokenized_text must be iterable")
        
        if len(tokenized_list) != len(text_list):
            raise ValueError(
                f"tokenized_text length ({len(tokenized_list)}) must match "
                f"text length ({len(text_list)})"
            )
    
    # Initialize models and params dicts
    if models is None:
        models = {}
    
    if embedding_params is None:
        embedding_params = {}
    
    embeddings = {}
    
    # Helper function for verbose printing
    def vprint(message):
        if verbose:
            print(f"[Embeddings] {message}")
    
    try:
        # 1. TF-IDF vectorization
        vprint("Generating TF-IDF vectors...")
        tfidf_params = embedding_params.get('tfidf', {})
        tfidf_matrix, vectorizer, vocab_size = vectorize_text(text_list, **tfidf_params)
        embeddings['tfidf'] = tfidf_matrix
        models['vectorizer'] = vectorizer
        vprint(f" TF-IDF complete. Shape: {tfidf_matrix.shape}, Vocab size: {vocab_size}")
        
        # 2. UMAP embeddings
        vprint("Generating UMAP embeddings...")
        umap_params = embedding_params.get('umap', {})
        embeddings['umap'], models['umap'] = get_umap_embeddings(tfidf_matrix, **umap_params)
        vprint(f"UMAP complete. Shape: {embeddings['umap'].shape}")
        
        # 3. t-SNE embeddings
        vprint("Generating t-SNE embeddings (this may take a while)...")
        tsne_params = embedding_params.get('tsne', {})
        embeddings['tsne'], models['tsne'] = get_tsne_embeddings(tfidf_matrix, **tsne_params)
        vprint(f"t-SNE complete. Shape: {embeddings['tsne'].shape}")
        
        # 4. LSA embeddings
        vprint("Generating LSA embeddings...")
        lsa_params = embedding_params.get('lsa', {})
        embeddings['lsa'], models['lsa'] = get_lsa_embeddings(tfidf_matrix, **lsa_params)
        vprint(f"LSA complete. Shape: {embeddings['lsa'].shape}")
        
        # 5. SBERT embeddings
        vprint("Generating SBERT embeddings...")
        sbert_model = models.get('sbert')
        sbert_params = embedding_params.get('sbert', {})
        embeddings['sbert'], models['sbert'] = get_sbert_embeddings(
            text_list, model=sbert_model, **sbert_params
        )
        vprint(f"SBERT complete. Shape: {embeddings['sbert'].shape}")
        
        # 6. Word-level embeddings (if tokenized text provided)
        if tokenized_text is not None:
            # Word2Vec
            vprint("Generating Word2Vec embeddings...")
            word2vec_params = embedding_params.get('word2vec', {})
            embeddings['word2vec'], models['word2vec'] = get_word2vec_embeddings(
                tokenized_list, **word2vec_params
            )
            vprint(f"Word2Vec complete. Shape: {embeddings['word2vec'].shape}")
            
            # GloVe (pretrained)
            vprint("Generating GloVe embeddings...")
            glove_model = models.get('glove')
            glove_params = embedding_params.get('glove', {})
            embeddings['glove'], models['glove'] = get_pretrained_word_embeddings(
                tokenized_list, model=glove_model, **glove_params
            )
            vprint(f"GloVe complete. Shape: {embeddings['glove'].shape}")
        else:
            vprint("Skipping Word2Vec and GloVe (no tokenized_text provided)")
        
        vprint(f"All embeddings generated successfully! Total: {len(embeddings)} types")
        
        return embeddings, models
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error generating embeddings: {str(e)}\n"
            f"Partial results may be available in the embeddings dict.",
            RuntimeWarning
        )
        # Return partial results
        return embeddings, models