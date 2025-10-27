# topic_extraction.py
"""
Topic Extraction and Keyword Analysis Module

This module provides tools for extracting keywords from clustered text data and performing
topic modeling using LDA (Latent Dirichlet Allocation) combined with BERT embeddings for
semantic matching with predefined designators.

Classes:
    KeywordExtractor: Extracts keywords from clustered narratives using multiple methods
    TopicModeler: Performs topic modeling and matches topics to designators
"""

import pandas as pd
from collections import defaultdict
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

# Import missing dependencies from utilities
from utilities import (
    calculate_tf_corpus,
    calculate_tfidf_corpus,
    calculate_tfdf_corpus,
    extract_kw_yake_corpus
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class KeywordExtractor:
    """
    Handles keyword extraction using multiple methods for clustered data.
    
    This class provides functionality to extract keywords from clustered narrative data
    using various statistical and machine learning methods including Term Frequency (TF),
    TF-IDF, TF-DF (Term Frequency-Document Frequency), and YAKE (Yet Another Keyword Extractor).
    
    Attributes:
        frame (pd.DataFrame): Input DataFrame containing clusters and narratives
        cluster_col (str): Name of the column containing cluster assignments
        narrative_col (str): Name of the column containing short narratives
        narrative_long_col (str): Name of the column containing long narratives
        keyframe (pd.DataFrame): Resulting DataFrame with extracted keywords per cluster
    """
    
    def __init__(self, 
                 frame: pd.DataFrame,
                 cluster_col: str = 'Clusters',
                 narrative_col: str = 'Narratives',
                 narrative_long_col: str = 'Narrative_long'):
        """
        Initialize the KeywordExtractor with a DataFrame containing clusters and narratives.
        
        Args:
            frame (pd.DataFrame): DataFrame with cluster and narrative columns. Must contain
                the columns specified by cluster_col, narrative_col, and narrative_long_col.
            cluster_col (str, optional): Name of the cluster column. Defaults to 'Clusters'.
            narrative_col (str, optional): Name of the short narrative column. 
                Defaults to 'Narratives'.
            narrative_long_col (str, optional): Name of the long narrative column. 
                Defaults to 'Narrative_long'.
        
        Raises:
            ValueError: If frame is None, empty, or missing required columns.
        """
        self._validate_dataframe(frame, cluster_col, narrative_col, narrative_long_col)
        self.frame = frame
        self.cluster_col = cluster_col
        self.narrative_col = narrative_col
        self.narrative_long_col = narrative_long_col
        self.keyframe = None
    
    def _validate_dataframe(self, frame: pd.DataFrame, cluster_col: str, 
                           narrative_col: str, narrative_long_col: str):
        """
        Validate that the DataFrame has all required columns and is not empty.
        
        Args:
            frame (pd.DataFrame): DataFrame to validate
            cluster_col (str): Expected cluster column name
            narrative_col (str): Expected short narrative column name
            narrative_long_col (str): Expected long narrative column name
        
        Raises:
            ValueError: If frame is None, empty, or missing any required columns.
        """
        if frame is None or frame.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        required_cols = [cluster_col, narrative_col, narrative_long_col]
        missing_cols = [col for col in required_cols if col not in frame.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    def extract_keywords_per_cluster(self, 
                                   tf_top_n: int = 5, 
                                   yake_top_n: int = 10, 
                                   yake_final_n: int = 5) -> pd.DataFrame:
        """
        Extract keywords for each cluster using multiple keyword extraction methods.
        
        This method applies five different keyword extraction techniques to each cluster:
        1. TF (Term Frequency)
        2. TF-IDF (Term Frequency-Inverse Document Frequency)
        3. TF-DF (Term Frequency-Document Frequency)
        4. YAKE on long narratives
        5. YAKE on short narratives
        
        Args:
            tf_top_n (int, optional): Number of top terms to extract for TF/TFIDF/TFDF methods. 
                Defaults to 5.
            yake_top_n (int, optional): Initial number of YAKE keywords to extract before filtering. 
                Defaults to 10.
            yake_final_n (int, optional): Final number of YAKE keywords to keep after filtering. 
                Defaults to 5.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['cluster', 'Yake Long', 'Yake Short', 
                'TF', 'TFIDF', 'TFDF'], where each cell contains comma-separated keywords.
        
        """
        tf_list, tfidf_list, tfdf_list = [], [], []
        yake_list_long, yake_list_short = [], []
        cluster_labels = []
        
        unique_clusters = self.frame[self.cluster_col].unique()
        logger.info(f"Extracting keywords for {len(unique_clusters)} clusters")
        
        for cluster in unique_clusters:
            cluster_filter = (self.frame[self.cluster_col] == cluster)
            cluster_frame = self.frame.loc[cluster_filter].copy()
            
            short_texts = cluster_frame[self.narrative_col]
            long_texts = cluster_frame[self.narrative_long_col]
            
            # Extract keywords using different methods
            tf_keywords = calculate_tf_corpus(short_texts, tf_top_n)
            tfidf_keywords = calculate_tfidf_corpus(short_texts, tf_top_n)
            tfdf_keywords = calculate_tfdf_corpus(short_texts, tf_top_n)
            yake_keywords_long = extract_kw_yake_corpus(long_texts, yake_top_n, yake_final_n)
            yake_keywords_short = extract_kw_yake_corpus(short_texts, yake_top_n, yake_final_n)
            
            tf_list.append(list(tf_keywords.keys()))
            tfidf_list.append(list(tfidf_keywords.keys()))
            tfdf_list.append(list(tfdf_keywords.keys()))
            yake_list_long.append(list(yake_keywords_long.keys()))
            yake_list_short.append(list(yake_keywords_short.keys()))
            cluster_labels.append(cluster)
        
        # Convert keyword lists to strings
        keyword_data = {
            'cluster': cluster_labels,
            'Yake Long': [", ".join(kw) for kw in yake_list_long],
            'Yake Short': [", ".join(kw) for kw in yake_list_short],
            'TF': [", ".join(kw) for kw in tf_list],
            "TFIDF": [", ".join(kw) for kw in tfidf_list],
            "TFDF": [", ".join(kw) for kw in tfdf_list]
        }
        
        self.keyframe = pd.DataFrame(keyword_data)
        logger.info(f"Keyword extraction complete")
        return self.keyframe
    
    def save_keywords(self, filepath: str):
        """
        Save the extracted keywords to a CSV file.
        
        Args:
            filepath (str): Path where the CSV file should be saved. Should include
                the .csv extension.
        
        Raises:
            ValueError: If no keywords have been extracted yet (keyframe is None).
        """
        if self.keyframe is None:
            raise ValueError("No keywords extracted yet. Call extract_keywords_per_cluster() first.")
        
        self.keyframe.to_csv(filepath, index=False)
        logger.info(f"Keywords saved to {filepath}")


class TopicModeler:
    """
    Handles topic modeling and topic-designator matching using LDA and BERT.
    
    This class provides comprehensive topic modeling capabilities by combining
    Latent Dirichlet Allocation (LDA) for discovering topics in text data with
    BERT (Bidirectional Encoder Representations from Transformers) embeddings
    for semantic matching between discovered topics and predefined designators.
    
    Attributes:
        texts (List[List[str]]): List of tokenized documents
        clusters (List[int]): Cluster assignment for each document
        dictionary (gensim.corpora.Dictionary): Gensim dictionary mapping words to IDs
        corpus (List): Bag-of-words representation of documents
        lda_model (gensim.models.LdaModel): Trained LDA model
        designators (Dict[str, str]): Dictionary of designator names and descriptions
        tokenizer (BertTokenizer): BERT tokenizer for text encoding
        bert_model (BertModel): Pre-trained BERT model for embeddings
        device (torch.device): Computing device (CPU or GPU)
        embedding_cache (Dict): Cache for storing computed embeddings
    """
    
    def __init__(self, 
                 texts: List[List[str]], 
                 clusters: List[int],
                 designators: Optional[Dict[str, str]] = None,
                 use_gpu: bool = True):
        """
        Initialize the TopicModeler with tokenized texts and cluster assignments.
        
        Args:
            texts (List[List[str]]): List of tokenized documents. Each document should
                be a list of tokens (words).
            clusters (List[int]): Cluster assignment for each document. Must be the
                same length as texts.
            designators (Optional[Dict[str, str]], optional): Dictionary mapping designator
                names to their descriptions. If None, uses default aviation safety designators.
                Defaults to None.
            use_gpu (bool, optional): Whether to use GPU acceleration for BERT computations
                if available. Defaults to True.
        
        Raises:
            ValueError: If texts or clusters are empty, or if their lengths don't match.
        """
        self._validate_inputs(texts, clusters)
        self.texts = texts
        self.clusters = clusters
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.designators = designators if designators else self._default_designators()
        self.tokenizer = None
        self.bert_model = None
        self.device = self._setup_device(use_gpu)
        self.embedding_cache = {}
    
    def _validate_inputs(self, texts: List[List[str]], clusters: List[int]):
        """
        Validate that input texts and clusters are properly formatted and aligned.
        
        Args:
            texts (List[List[str]]): List of tokenized documents to validate
            clusters (List[int]): List of cluster assignments to validate
        
        Raises:
            ValueError: If texts is empty, clusters is empty, or their lengths don't match.
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        if not clusters:
            raise ValueError("clusters cannot be empty")
        if len(texts) != len(clusters):
            raise ValueError(f"Length mismatch: texts ({len(texts)}) vs clusters ({len(clusters)})")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """
        Configure the computing device for PyTorch operations (GPU or CPU).
        
        Args:
            use_gpu (bool): Whether to attempt using GPU if available
        
        Returns:
            torch.device: The configured device (either 'cuda' or 'cpu')
        """
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using GPU for BERT computations")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for BERT computations")
        return device
    
    def _default_designators(self) -> Dict[str, str]:
        """
        Return the default set of aviation safety designators with descriptions.
        
        Returns:
            Dict[str, str]: Dictionary mapping designator names to their detailed descriptions.
                Contains 10 aviation safety-related designators covering knowledge adequacy,
                judgment, procedures, communication, monitoring, task management, stress,
                physiological factors, technical failures, and environmental factors.
        
        Note:
            These designators are specifically designed for aviation safety incident analysis
            but can be customized by providing a custom designators dictionary during initialization.
        """
        return {
            "Inadequate or inaccurate knowledge (Adequacy of knowledge)": 
                "Inadequate inaccurate knowledge, Aeronautical Knowledge, Systems Knowledge, "
                "Procedural Knowledge, Technical Understanding",
            
            "Poor judgment and decision-making": 
                "Poor judgment, Decision-making errors, Risk assessment failures, "
                "Inadequate situation evaluation",
            
            "Failure to follow procedures": 
                "Non-compliance with procedures, Procedural violations, Checklist omissions, "
                "Standard Operating Procedure deviations",
            
            "Poor communication": 
                "Communication breakdowns, Inadequate crew coordination, "
                "Misunderstandings, Information transfer failures",
            
            "Inadequate monitoring or vigilance": 
                "Lack of monitoring, Reduced vigilance, Insufficient cross-checking, "
                "Failure to detect changes",
            
            "Task management and prioritization": 
                "Poor task prioritization, Workload management issues, "
                "Task saturation, Resource allocation problems",
            
            "Stress and psychological factors": 
                "Stress impact, Psychological pressure, Emotional factors, "
                "Mental state issues, Fatigue effects",
            
            "Physical or physiological factors": 
                "Physical limitations, Physiological impairment, Health issues, "
                "Physical fatigue, Medical conditions",
            
            "Technical or system failures": 
                "Equipment failures, System malfunctions, Technical problems, "
                "Mechanical issues, Instrumentation failures",
            
            "Environmental factors": 
                "Weather conditions, Environmental challenges, External conditions, "
                "Atmospheric factors, Terrain issues"
        }
    
    def prepare_corpus(self):
        """
        Create a Gensim dictionary and bag-of-words corpus from the tokenized texts.
        
        This method must be called before training the LDA model. It converts the
        tokenized texts into the format required by Gensim's LDA implementation.
        """
        logger.info("Preparing corpus for LDA")
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        logger.info(f"Dictionary size: {len(self.dictionary)}, Corpus size: {len(self.corpus)}")
    
    def train_lda(self, num_topics: int = 15, passes: int = 15, random_state: int = 42):
        """
        Train a Latent Dirichlet Allocation (LDA) model on the prepared corpus.
        
        LDA is a generative probabilistic model that discovers abstract topics in
        a collection of documents. Each document is modeled as a mixture of topics,
        and each topic is modeled as a mixture of words.
        
        Args:
            num_topics (int, optional): Number of latent topics to extract from the corpus.
                Defaults to 15.
            passes (int, optional): Number of passes through the entire corpus during training.
                More passes can improve model quality but increase training time. Defaults to 15.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        if self.corpus is None:
            self.prepare_corpus()
        
        logger.info(f"Training LDA model with {num_topics} topics")
        self.lda_model = LdaModel(
            corpus=self.corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=passes,
            random_state=random_state
        )
        logger.info("LDA training complete")
    
    def get_cluster_topics(self, top_n: int = 3) -> Dict[int, List[int]]:
        """
        Identify the most probable topics for each cluster based on document-topic distributions.
        
        For each cluster, this method computes the average topic distribution across all
        documents in that cluster and returns the top N most probable topics.
        
        Args:
            top_n (int, optional): Number of top topics to return for each cluster. 
                Defaults to 3.
        
        Returns:
            Dict[int, List[int]]: Dictionary mapping cluster IDs to lists of topic IDs,
                ordered by probability (highest first).
        
        Raises:
            ValueError: If the LDA model has not been trained yet.
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet. Call train_lda() first.")
        
        cluster_topics = {}
        unique_clusters = set(self.clusters)
        
        logger.info(f"Computing topics for {len(unique_clusters)} clusters")
        
        for cluster_id in unique_clusters:
            # Get indices of texts in this cluster
            cluster_indices = [i for i, c in enumerate(self.clusters) if c == cluster_id]
            
            # Compute average topic probabilities incrementally
            topic_sums = defaultdict(float)
            count = 0
            
            for idx in cluster_indices:
                bow = self.corpus[idx]
                topic_probs = self.lda_model.get_document_topics(bow)
                
                for topic_id, prob in topic_probs:
                    topic_sums[topic_id] += prob
                count += 1
            
            # Calculate averages
            topic_avg_probs = {
                topic_id: total / count 
                for topic_id, total in topic_sums.items()
            }
            
            # Get top topics
            top_topics = sorted(topic_avg_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
            cluster_topics[cluster_id] = [topic_id for topic_id, _ in top_topics]
        
        return cluster_topics
    
    def load_bert(self, model_name: str = 'bert-base-uncased'):
        """
        Load a pre-trained BERT model and tokenizer for computing text embeddings.
        
        Args:
            model_name (str, optional): Name of the pre-trained BERT model to load from
                the Hugging Face model hub. Defaults to 'bert-base-uncased'.
        """
        if self.tokenizer is None or self.bert_model is None:
            logger.info(f"Loading BERT model: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = BertModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            logger.info("BERT model loaded successfully")
    
    def get_bert_embedding(self, text: str, use_cache: bool = True, max_length: int = 512) -> np.ndarray:
        """
        Compute a BERT embedding vector for the given text.
        
        This method tokenizes the input text, passes it through the BERT model,
        and returns the embedding of the [CLS] token as a dense vector representation.
        
        Args:
            text (str): Input text to embed
            use_cache (bool, optional): Whether to use cached embeddings if available
                and cache new embeddings for future use. Defaults to True.
            max_length (int, optional): Maximum sequence length for BERT tokenization.
                Sequences longer than this will be truncated. Defaults to 512.
        
        Returns:
            np.ndarray: A 1D numpy array containing the embedding vector (typically 768 dimensions
                for base BERT models).
        
        """
        # Check cache first
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Load BERT if not already loaded
        if self.tokenizer is None or self.bert_model is None:
            self.load_bert()
        
        # Generate embedding
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Cache the result
        if use_cache:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    def get_topic_words(self, topic_id: int, num_words: int = 10) -> List[str]:
        """
        Retrieve the top words associated with a specific topic from the LDA model.
        
        Args:
            topic_id (int): ID of the topic (must be in range [0, num_topics-1])
            num_words (int, optional): Number of top words to return. Defaults to 10.
        
        Returns:
            List[str]: List of the most probable words for the specified topic,
                ordered by probability (highest first).
        
        Raises:
            ValueError: If the LDA model has not been trained yet.
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet. Call train_lda() first.")
        
        return [word for word, _ in self.lda_model.show_topic(topic_id, num_words)]
    
    def match_designators_to_topics(self, 
                                   num_words: int = 10, 
                                   top_k: int = 3) -> Dict[int, List[Tuple[str, float]]]:
        """
        Match predefined designators to discovered topics using BERT embedding similarity.
        
        This method computes semantic similarity between topics (represented by their top words)
        and designators (represented by their descriptions) using BERT embeddings and cosine
        similarity. This allows for meaningful interpretation of discovered topics.
        
        Args:
            num_words (int, optional): Number of top words to use for representing each topic.
                Defaults to 10.
            top_k (int, optional): Number of top matching designators to return for each topic.
                Defaults to 3.
        
        Returns:
            Dict[int, List[Tuple[str, float]]]: Dictionary mapping topic IDs to lists of
                (designator_name, similarity_score) tuples, ordered by similarity (highest first).
        
        Raises:
            ValueError: If the LDA model has not been trained yet.
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet. Call train_lda() first.")
        
        logger.info("Computing designator-topic matches using BERT embeddings")
        
        # Get embeddings for all designators (with caching)
        designator_list = list(self.designators.keys())
        designator_embeddings = np.vstack([
            self.get_bert_embedding(self.designators[designator], use_cache=True)
            for designator in designator_list
        ])
        
        # Get embeddings for all topics and compute similarities
        topic_designators = {}
        for topic_id in range(self.lda_model.num_topics):
            topic_words = self.get_topic_words(topic_id, num_words)
            topic_text = " ".join(topic_words)
            topic_embedding = self.get_bert_embedding(topic_text, use_cache=True)
            
            # Vectorized similarity computation
            similarities = cosine_similarity(topic_embedding, designator_embeddings)[0]
            
            # Get top k designators
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            topic_designators[topic_id] = [
                (designator_list[idx], similarities[idx]) 
                for idx in top_indices
            ]
        
        logger.info("Designator matching complete")
        return topic_designators
    
    def clear_cache(self):
        """
        Clear the embedding cache to free memory.
        
        The embedding cache can grow large when processing many unique texts.
        Use this method to free up memory when the cache is no longer needed
        or before processing a new batch of data.
        
        Note:
            After clearing the cache, subsequent calls to get_bert_embedding()
            will need to recompute embeddings for previously cached texts.
        """
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")