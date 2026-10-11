from typing import Optional, Dict, Any, List, Union
import logging
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
try:
    from .preprocessing import pre_processing_routine, remove_stopwords_corpus, define_stopwords
    from .clustering_tuning import full_clustering_pipeline, full_clustering_pipeline_fixed_params
    from .clustering_tuning import DEFAULT_UMAP_GRID, DEFAULT_HDBSCAN_GRID, DEFAULT_MAX_PAIRWISE_MEMORY_MB
    from .visualization import plot_clusters, plot_clusters_with_annotations
    from .embeddings import vectorize_text
except ImportError:
    from preprocessing import pre_processing_routine, remove_stopwords_corpus, define_stopwords
    from clustering_tuning import full_clustering_pipeline, full_clustering_pipeline_fixed_params
    from clustering_tuning import DEFAULT_UMAP_GRID, DEFAULT_HDBSCAN_GRID, DEFAULT_MAX_PAIRWISE_MEMORY_MB
    from visualization import plot_clusters, plot_clusters_with_annotations
    from embeddings import vectorize_text

class _LazyTopicClass:
    """Resolve topic classes only when topic functionality is requested."""
    class_name = None

    def __new__(cls, *args, **kwargs):
        try:
            from . import topic_extraction
        except ImportError:
            import topic_extraction
        implementation = getattr(topic_extraction, cls.class_name)
        return implementation(*args, **kwargs)


class KeywordExtractor(_LazyTopicClass):
    class_name = "KeywordExtractor"


class TopicModeler(_LazyTopicClass):
    class_name = "TopicModeler"


def _resolve_umap_metric(metric, umap_metric):
    """Keep the legacy metric argument as a UMAP-only alias."""
    if metric is not None and umap_metric is not None and metric != umap_metric:
        raise ValueError("metric and umap_metric must agree when both are supplied")
    return umap_metric if umap_metric is not None else (metric if metric is not None else 'cosine')


class ClusteringPipeline:
    """
    End-to-end clustering pipeline wrapper with modular components.
    Optimized for memory efficiency and performance.
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str = 'Narrative_long', random_state: int = 42):
        """
        Initialize the pipeline with a DataFrame and text column.
        
        Args:
            df: Input DataFrame (not copied until modification)
            text_column: Name of column containing text to analyze
            random_state: Random state for reproducibility in all stochastic operations
        """
        self.df = df  # Don't copy until we actually modify
        self.text_column = text_column
        self.random_state = random_state
        self._df_modified = False
        self._vectorized_text = None  # Cache vectorization
        self._stopwords = None  # Cache stopwords
        self._stopwords_params_key = None
        self._generated_columns = set()
        self.clustering_results = None
        self.keywords = None
        self.topics = None
        
    def _ensure_df_copy(self):
        """Ensure DataFrame is copied before modification."""
        if not self._df_modified:
            self.df = self.df.copy()
            self._df_modified = True

    def _invalidate_analysis_state(self):
        """Clear cached results and columns derived from an earlier analysis."""
        self._vectorized_text = None
        self._stopwords = None
        self._stopwords_params_key = None
        self.clustering_results = None
        self.keywords = None
        self.topics = None
        if self._generated_columns:
            self.df.drop(columns=list(self._generated_columns), inplace=True, errors='ignore')
            self._generated_columns.clear()

    def _invalidate_clustering_state(self):
        """Discard results and derived labels before a new clustering run."""
        self.clustering_results = None
        self.keywords = None
        self.topics = None
        if self._generated_columns or 'Clusters' in self.df.columns:
            self._ensure_df_copy()
        columns_to_clear = set(self._generated_columns)
        columns_to_clear.add('Clusters')
        self.df.drop(columns=list(columns_to_clear), inplace=True, errors='ignore')
        self._generated_columns.clear()
    
    def _get_vectorized_text(self, force_recompute: bool = False) -> Union[np.ndarray, spmatrix]:
        """
        Get or compute vectorized text, with caching.
        
        Args:
            force_recompute: Force recomputation even if cached
            
        Returns:
            Cached feature matrix (sparse CSR for the default TF-IDF vectorizer).
        """
        if self._vectorized_text is None or force_recompute:
            if 'processed_stopword_Narr' not in self.df.columns:
                raise ValueError(
                    "Data must be preprocessed first. Call preprocess_data() before clustering."
                )
            
            # Call vectorize_text and handle if it returns a tuple
            result = vectorize_text(self.df['processed_stopword_Narr'].values)
            
            # Handle case where vectorize_text returns (X, vectorizer) tuple
            if isinstance(result, tuple):
                self._vectorized_text = result[0]  # Get just the matrix
            else:
                self._vectorized_text = result
                
        return self._vectorized_text
        
    def preprocess_data(self, 
                       custom_stopwords: Optional[List[str]] = None,
                       keep_words: Optional[List[str]] = None,
                       low_idf: bool = True,
                       idf_threshold: float = 1.4) -> pd.DataFrame:
        """
        Preprocess the text data with standard cleaning pipeline.
        
        Args:
            custom_stopwords: Additional domain-specific stopwords
            keep_words: Words to never remove as stopwords
            low_idf: Whether to remove low IDF words
            idf_threshold: IDF threshold for word removal
            
        Returns:
            DataFrame with processed text columns
        """
        # Only copy if we haven't already
        self._ensure_df_copy()

        # Reprocessing may change every downstream result, so discard state
        # produced by the previous preprocessing/clustering pass first.
        self._invalidate_analysis_state()
        
        # Run preprocessing routine
        self.df['processed'], self.df['processed_num_Narr'] = pre_processing_routine(self.df, text_column=self.text_column)

        # Build a key representing the current stopword parameters so we can
        # detect when the caller has changed them and must recompute the cache
        stopword_params_key = (
            tuple(sorted(custom_stopwords or [])),
            tuple(sorted(keep_words or [])),
            low_idf,
            idf_threshold
        )

        # Recompute stopwords if not yet cached or if parameters have changed
        if self._stopwords is None or getattr(self, '_stopwords_params_key', None) != stopword_params_key:
            self._stopwords = define_stopwords(
                self.df,
                text_column='processed_num_Narr',
                custom_stop_words=custom_stopwords or [],
                custom_keep_words=keep_words or [],
                low_idf=low_idf,
                idf_threshold=idf_threshold
            )
            self._stopwords_params_key = stopword_params_key
        
        # Remove stopwords
        self.df['processed_stopword_Narr'] = remove_stopwords_corpus(
            self.df['processed_num_Narr'].values, 
            self._stopwords
        )
        
        # Invalidate cached vectorization since data changed
        self._vectorized_text = None
        
        return self.df
    
    def cluster_data(self,
                n_neighbors: int = 10,
                min_dist: float = 0.0,
                metric: Optional[str] = None,
                n_components: int = 5,
                min_cluster_size: int = 10,
                cluster_selection_method: str = 'eom',
                standardize: bool = False,
                compute_tsne: bool = False,
                tsne_perplexity: int = 30,
                tsne_early_exaggeration: int = 12,
                tsne_learning_rate: int = 200,
                random_state: Optional[int] = None,
                *,
                umap_metric: Optional[str] = None,
                hdbscan_metric: str = 'euclidean',
                max_pairwise_memory_mb: float = DEFAULT_MAX_PAIRWISE_MEMORY_MB) -> Dict[str, Any]:
        """
        Perform clustering with specified parameters (no tuning).
    
        Args:
            n_neighbors: UMAP neighborhood size
            min_dist: UMAP minimum distance
            metric: Legacy alias for umap_metric; no longer sets HDBSCAN's metric.
            umap_metric: UMAP input distance, default 'cosine'.
            hdbscan_metric: Distance on reduced coordinates, default 'euclidean'.
            max_pairwise_memory_mb: Cosine distance-matrix limit in MiB, excluding workspace.
            n_components: Number of UMAP components
            min_cluster_size: HDBSCAN minimum cluster size
            cluster_selection_method: HDBSCAN cluster selection method
            standardize: Opt-in feature scaling; sparse TF-IDF is never centered.
            compute_tsne: Opt-in t-SNE computation for visualization.
            tsne_perplexity: t-SNE perplexity
            tsne_early_exaggeration: t-SNE early exaggeration
            tsne_learning_rate: t-SNE learning rate
            random_state: Random state for reproducibility. If None, uses instance random_state
        
        Returns:
            Dictionary containing clustering results
        """
        # Get vectorized text (uses cache if available)
        umap_metric = _resolve_umap_metric(metric, umap_metric)
        self._invalidate_clustering_state()
        X = self._get_vectorized_text()
        
        # Use instance random state if not provided
        rs = random_state if random_state is not None else self.random_state
    
        # Call fixed params pipeline
        self.clustering_results = full_clustering_pipeline_fixed_params(
            X=X,
            umap_params={
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'metric': umap_metric,
                'n_components': n_components
            },
            hdbscan_params={
                'min_cluster_size': min_cluster_size,
                'metric': hdbscan_metric,
                'cluster_selection_method': cluster_selection_method
            },
            tsne_params={
                'perplexity': tsne_perplexity,
                'early_exaggeration': tsne_early_exaggeration,
                'learning_rate': tsne_learning_rate,
                'n_components': 2,
                'random_state': rs
            },
            standardize=standardize,
            compute_tsne=compute_tsne,
            progress=True,
            random_state=rs,
            max_pairwise_memory_mb=max_pairwise_memory_mb
        )
    
        # Add clusters to DataFrame
        if (self.clustering_results.get('clustering_results') is not None and 
            self.clustering_results['clustering_results'] is not None):
            self._ensure_df_copy()
            self.df['Clusters'] = self.clustering_results['clustering_results']['clusters']
            self._generated_columns.add('Clusters')
    
        return self.clustering_results
    
    def cluster_and_analyze_topics(self,
                                  n_neighbors: int = 10,
                                  min_dist: float = 0.0,
                                  metric: Optional[str] = None,
                                  n_components: int = 5,
                                  min_cluster_size: int = 10,
                                  cluster_selection_method: str = 'eom',
                                  standardize: bool = False,
                                  compute_tsne: bool = False,
                                  tsne_perplexity: int = 30,
                                  tsne_early_exaggeration: int = 12,
                                  tsne_learning_rate: int = 200,
                                  num_topics: int = 15,
                                  passes: int = 15,
                                  designators: Optional[Dict] = None,
                                  designator_profile: Optional[str] = None,
                                  min_similarity: float = 0.5,
                                  random_state: Optional[int] = None,
                                  *,
                                  umap_metric: Optional[str] = None,
                                  hdbscan_metric: str = 'euclidean',
                                  max_pairwise_memory_mb: float = DEFAULT_MAX_PAIRWISE_MEMORY_MB) -> Dict[str, Any]:
        """
        Perform clustering and LDA topic modeling in one integrated step.
        
        This method combines clustering and topic analysis using LDA (Latent Dirichlet Allocation)
        to discover the main topics within each cluster.
        
        Args:
            n_neighbors: UMAP neighborhood size
            min_dist: UMAP minimum distance
            metric: Legacy alias for umap_metric; no longer sets HDBSCAN's metric.
            umap_metric: UMAP input distance, default 'cosine'.
            hdbscan_metric: Distance on reduced coordinates, default 'euclidean'.
            max_pairwise_memory_mb: Cosine distance-matrix limit in MiB, excluding workspace.
            n_components: Number of UMAP components
            min_cluster_size: HDBSCAN minimum cluster size
            cluster_selection_method: HDBSCAN cluster selection method
            standardize: Whether to standardize data before clustering
            compute_tsne: Whether to compute t-SNE embeddings for visualization
            tsne_perplexity: t-SNE perplexity
            tsne_early_exaggeration: t-SNE early exaggeration
            tsne_learning_rate: t-SNE learning rate
            num_topics: Number of LDA topics to extract
            passes: Number of LDA training passes
            designators: Optional dictionary for topic-to-designator matching
            designator_profile: Optional built-in domain profile, currently 'aviation'.
                Omit for domain-neutral topics that may remain unmatched.
            min_similarity: Minimum cosine similarity required to assign a designator.
            random_state: Random state for reproducibility. If None, uses instance random_state
            
        Returns:
            Dictionary containing clustering and topic analysis results:
                - 'clustering_results': Full clustering output
                - 'topics': Dictionary with LDA topic analysis
                - 'cluster_summary': Summary statistics per cluster
                - 'dataframe': Updated DataFrame with cluster assignments
        """
        # Use instance random state if not provided
        rs = random_state if random_state is not None else self.random_state
        
        print("Step 1/3: Clustering data...")
        
        # Perform clustering
        clustering_results = self.cluster_data(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method,
            standardize=standardize,
            compute_tsne=compute_tsne,
            tsne_perplexity=tsne_perplexity,
            tsne_early_exaggeration=tsne_early_exaggeration,
            tsne_learning_rate=tsne_learning_rate,
            random_state=rs,
            umap_metric=umap_metric,
            hdbscan_metric=hdbscan_metric,
            max_pairwise_memory_mb=max_pairwise_memory_mb
        )
        
        print("Step 2/3: Performing LDA topic modeling...")
        
        # Perform topic analysis
        topics = self.analyze_topics(
            num_topics=num_topics,
            passes=passes,
            designators=designators,
            designator_profile=designator_profile,
            min_similarity=min_similarity,
            random_state=rs
        )
        
        print("Step 3/3: Generating cluster summary...")
        
        # Generate cluster summary
        cluster_summary = self.get_cluster_summary()
        
        # Combine results
        combined_results = {
            'clustering_results': clustering_results,
            'topics': topics,
            'cluster_summary': cluster_summary,
            'dataframe': self.df
        }
        
        print("\nClustering and topic analysis complete!")
        print(f"Found {len(cluster_summary)} clusters")
        print(f"Extracted {num_topics} topics using LDA")
        print(f"Largest cluster: {cluster_summary.iloc[0]['Size']} documents ({cluster_summary.iloc[0]['Percentage']}%)")
        
        return combined_results

    def cluster_and_extract_keywords(self,
                                    n_neighbors: int = 10,
                                    min_dist: float = 0.0,
                                    metric: Optional[str] = None,
                                    n_components: int = 5,
                                    min_cluster_size: int = 10,
                                    cluster_selection_method: str = 'eom',
                                    standardize: bool = False,
                                    compute_tsne: bool = False,
                                    tsne_perplexity: int = 30,
                                    tsne_early_exaggeration: int = 12,
                                    tsne_learning_rate: int = 200,
                                    tf_top_n: int = 5,
                                    yake_top_n: int = 10,
                                    yake_final_n: int = 5,
                                    random_state: Optional[int] = None,
                                    *,
                                    umap_metric: Optional[str] = None,
                                    hdbscan_metric: str = 'euclidean',
                                    max_pairwise_memory_mb: float = DEFAULT_MAX_PAIRWISE_MEMORY_MB) -> Dict[str, Any]:
        """
        Perform clustering and keyword extraction in one integrated step.
        
        This method combines clustering and keyword extraction for efficiency,
        automatically running both operations and returning combined results.
        
        Args:
            n_neighbors: UMAP neighborhood size
            min_dist: UMAP minimum distance
            metric: Legacy alias for umap_metric; no longer sets HDBSCAN's metric.
            umap_metric: UMAP input distance, default 'cosine'.
            hdbscan_metric: Distance on reduced coordinates, default 'euclidean'.
            max_pairwise_memory_mb: Cosine distance-matrix limit in MiB, excluding workspace.
            n_components: Number of UMAP components
            min_cluster_size: HDBSCAN minimum cluster size
            cluster_selection_method: HDBSCAN cluster selection method
            standardize: Whether to standardize data before clustering
            compute_tsne: Whether to compute t-SNE embeddings for visualization
            tsne_perplexity: t-SNE perplexity
            tsne_early_exaggeration: t-SNE early exaggeration
            tsne_learning_rate: t-SNE learning rate
            tf_top_n: Number of top terms for TF methods
            yake_top_n: Initial YAKE keywords to extract
            yake_final_n: Final YAKE keywords to keep
            random_state: Random state for reproducibility. If None, uses instance random_state
            
        Returns:
            Dictionary containing both clustering and keyword extraction results:
                - 'clustering_results': Full clustering output
                - 'keywords': DataFrame with keywords per cluster
                - 'cluster_summary': Summary statistics per cluster
        """
        # Use instance random state if not provided
        rs = random_state if random_state is not None else self.random_state
        
        print("Step 1/2: Clustering data...")
        
        # Perform clustering
        clustering_results = self.cluster_data(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method,
            standardize=standardize,
            compute_tsne=compute_tsne,
            tsne_perplexity=tsne_perplexity,
            tsne_early_exaggeration=tsne_early_exaggeration,
            tsne_learning_rate=tsne_learning_rate,
            random_state=rs,
            umap_metric=umap_metric,
            hdbscan_metric=hdbscan_metric,
            max_pairwise_memory_mb=max_pairwise_memory_mb
        )
        
        print("Step 2/2: Extracting keywords...")
        
        # Extract keywords
        keywords = self.extract_keywords(
            tf_top_n=tf_top_n,
            yake_top_n=yake_top_n,
            yake_final_n=yake_final_n
        )
        
        # Generate cluster summary
        cluster_summary = self.get_cluster_summary()
        
        # Combine results
        combined_results = {
            'clustering_results': clustering_results,
            'keywords': keywords,
            'cluster_summary': cluster_summary,
            'dataframe': self.df
        }
        
        print("\nClustering and keyword extraction complete!")
        print(f"Found {len(cluster_summary)} clusters")
        print(f"Largest cluster: {cluster_summary.iloc[0]['Size']} documents ({cluster_summary.iloc[0]['Percentage']}%)")
        
        return combined_results
    
    def tune_and_cluster(self,
                        min_clusters: int = 3,
                        time_limit: int = 600,
                        max_iterations: int = 50,
                        target_metric: str = 'davies_bouldin',
                        standardize: bool = False,
                        tune_tsne: bool = False,
                        random_state: Optional[int] = None,
                        *,
                        compute_tsne: bool = False,
                        tsne_perplexity: int = 30,
                        tsne_early_exaggeration: int = 12,
                        tsne_learning_rate: int = 200,
                        min_coverage: float = 0.5,
                        umap_metric: Optional[str] = None,
                        hdbscan_metric: str = 'euclidean',
                        metric: Optional[str] = None,
                        max_pairwise_memory_mb: float = DEFAULT_MAX_PAIRWISE_MEMORY_MB) -> Dict[str, Any]:
        """
        Perform clustering with hyperparameter tuning.
        
        Args:
            min_clusters: Minimum number of clusters required
            time_limit: Maximum tuning time in seconds
            max_iterations: Maximum tuning iterations
            target_metric: Metric to optimize ('davies_bouldin' or 'calinski_harabasz')
            standardize: Whether to standardize data before clustering
            tune_tsne: Unsupported; True raises ValueError because t-SNE is not scored.
            random_state: Random state for reproducibility. If None, uses instance random_state
            compute_tsne: Opt-in computation of final t-SNE coordinates.
            tsne_perplexity: Perplexity for fixed t-SNE parameters.
            tsne_early_exaggeration: Early exaggeration for fixed t-SNE parameters.
            tsne_learning_rate: Learning rate for fixed t-SNE parameters.
            umap_metric: UMAP input distance, default 'cosine'.
            hdbscan_metric: HDBSCAN distance, default 'euclidean'.
            metric: Legacy UMAP-only alias, consistent with cluster_data().
            max_pairwise_memory_mb: Cosine distance-matrix limit in MiB, excluding workspace.
            min_coverage: Minimum share of documents assigned to clusters for a tuning
                candidate to be eligible.
            
        Returns:
            Dictionary containing clustering results
        """
        # Use instance random state if not provided
        rs = random_state if random_state is not None else self.random_state

        if tune_tsne:
            raise ValueError(
                "tune_tsne is unsupported: clustering search evaluates UMAP/HDBSCAN only. "
                "Tune t-SNE separately for visualization."
            )
        
        umap_metric = _resolve_umap_metric(metric, umap_metric)
        self._invalidate_clustering_state()
        # Get vectorized text (uses cache if available)
        X = self._get_vectorized_text()
        
        # Run full tuning pipeline
        self.clustering_results = full_clustering_pipeline(
            X=X,
            standardize=standardize,
            min_clusters=min_clusters,
            time_limit=time_limit,
            max_iterations=max_iterations,
            target_metric=target_metric,
            progress=True,
            tune_tsne=tune_tsne,
            random_state=rs,
            compute_tsne=compute_tsne,
            tsne_params={
                'perplexity': tsne_perplexity,
                'early_exaggeration': tsne_early_exaggeration,
                'learning_rate': tsne_learning_rate,
                'n_components': 2,
                'random_state': rs
            },
            umap_param_grid=dict(DEFAULT_UMAP_GRID, metric=[umap_metric]),
            hdbscan_param_grid=dict(DEFAULT_HDBSCAN_GRID, metric=[hdbscan_metric]),
            max_pairwise_memory_mb=max_pairwise_memory_mb,
            min_coverage=min_coverage
        )
        
        # Add clusters to DataFrame
        if (self.clustering_results.get('clustering_results') is not None and 
            self.clustering_results['clustering_results'] is not None):
            self._ensure_df_copy()
            self.df['Clusters'] = self.clustering_results['clustering_results']['clusters']
            self._generated_columns.add('Clusters')
        
        return self.clustering_results
    
    def visualize_clusters(self,
                      show_outliers: bool = True,
                      annotation_column: Optional[str] = None,
                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Visualize the clustering results.
    
        Args:
            show_outliers: Whether to include outliers in visualization
            annotation_column: Column to use for point annotations
            save_path: Path to save visualization (optional)
        
        Returns:
            matplotlib Figure if showing plot, None if saving to file
        """
        if self.clustering_results is None:
            raise ValueError("No clustering results available. Run cluster_data() or tune_and_cluster() first.")
    
        if self.clustering_results.get('clustering_results') is None:
            raise ValueError("Clustering failed or produced no results.")
    
    
        embeddings = self.clustering_results['clustering_results']['tsne_embeddings']
        clusters = self.clustering_results['clustering_results']['clusters']
    
        if embeddings is None:
            # More detailed error message
            print("ERROR: t-SNE embeddings are None")
            print("Check if there was a warning during clustering about t-SNE computation failing")
            raise ValueError("No t-SNE embeddings available. Ensure compute_tsne=True when clustering.")
    
        if annotation_column:
            if annotation_column not in self.df.columns:
                raise ValueError(f"Annotation column '{annotation_column}' not found in DataFrame")
        
            return plot_clusters_with_annotations(
                embeddings=embeddings,
                labels=clusters,
                annotations=self.df[annotation_column].values,
                filter_outliers=not show_outliers,
                save_path=save_path
            )
        else:
            return plot_clusters(
                embeddings=embeddings,
                labels=clusters,
                show_outliers=show_outliers,
                save_path=save_path
            )
    
    def extract_keywords(self,
                        tf_top_n: int = 5,
                        yake_top_n: int = 10,
                        yake_final_n: int = 5) -> pd.DataFrame:
        """
        Extract keywords for each cluster using multiple methods.
        
        Args:
            tf_top_n: Number of top terms for TF methods
            yake_top_n: Initial YAKE keywords to extract
            yake_final_n: Final YAKE keywords to keep
            
        Returns:
            DataFrame with keywords per cluster per method
        """
        if 'Clusters' not in self.df.columns:
            raise ValueError("No cluster assignments found. Run clustering first.")
        
        # Use module-level import if available
        # KeywordExtractor expects both 'Narratives' and 'Narrative_long' columns
        # Create 'Narratives' column if it doesn't exist (use processed text or original)
        if 'Narratives' not in self.df.columns:
            self._ensure_df_copy()
            # Use processed text if available, otherwise use original narrative
            if 'processed_stopword_Narr' in self.df.columns:
                self.df['Narratives'] = self.df['processed_stopword_Narr']
            elif 'processed' in self.df.columns:
                self.df['Narratives'] = self.df['processed']
            else:
                self.df['Narratives'] = self.df[self.text_column]
            self._generated_columns.add('Narratives')
        
        extractor = KeywordExtractor(self.df, narrative_long_col=self.text_column)
        self.keywords = extractor.extract_keywords_per_cluster(
            tf_top_n=tf_top_n,
            yake_top_n=yake_top_n,
            yake_final_n=yake_final_n
        )
        return self.keywords
    
    def analyze_topics(self,
                      num_topics: int = 15,
                      passes: int = 15,
                      designators: Optional[Dict] = None,
                      designator_profile: Optional[str] = None,
                      min_similarity: float = 0.5,
                      random_state: Optional[int] = None) -> Dict:
        """
        Perform topic modeling and match topics to designators.
        
        Args:
            num_topics: Number of LDA topics
            passes: Number of LDA passes
            designators: Custom designator dictionary
            designator_profile: Optional built-in domain profile, currently 'aviation'.
                Omit for domain-neutral topics that may remain unmatched.
            min_similarity: Minimum cosine similarity required to assign a designator.
            random_state: Random state for reproducibility. If None, uses instance random_state
            
        Returns:
            Dictionary with topic analysis results
        """
        # Use instance random state if not provided
        rs = random_state if random_state is not None else self.random_state
        
        if 'Clusters' not in self.df.columns:
            raise ValueError("No cluster assignments found. Run clustering first.")
        
        # Check if preprocessing was done
        if 'processed_stopword_Narr' not in self.df.columns:
            raise ValueError("Data must be preprocessed first. Call preprocess_data().")

        # HDBSCAN's -1 label means noise, not a meaningful topic cluster.
        # Keep those documents out of both LDA fitting and cluster summaries.
        topic_rows = self.df['Clusters'] != -1
        if not topic_rows.any():
            raise ValueError("Topic analysis requires at least one non-noise cluster.")
        
        # Use module-level import if available
        # Tokenize using vectorized string operation (faster than apply)
        if 'Processed_Narrative' not in self.df.columns:
            self._ensure_df_copy()
            self.df['Processed_Narrative'] = self.df['processed_stopword_Narr'].str.split()
            self._generated_columns.add('Processed_Narrative')
        
        modeler = TopicModeler(
            texts=self.df.loc[topic_rows, 'Processed_Narrative'].tolist(),
            clusters=self.df.loc[topic_rows, 'Clusters'].tolist(),
            designators=designators,
            designator_profile=designator_profile
        )
        
        modeler.train_lda(num_topics=num_topics, passes=passes, random_state=rs)
        cluster_topics = modeler.get_cluster_topics()
        topic_designators = modeler.match_designators_to_topics(min_similarity=min_similarity)
        
        self.topics = {
            'cluster_topics': cluster_topics,
            'topic_designators': topic_designators,
            'topic_match_metadata': getattr(modeler, 'topic_match_metadata', {}),
            'model': modeler
        }
        
        return self.topics
    
    def full_pipeline(self,
                      tune_hyperparameters: bool = False,
                      visualize: bool = False,
                      extract_keywords: bool = True,
                      analyze_topics: bool = True,
                      min_clusters: int = 3,
                      time_limit: int = 600,
                      random_state: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Run complete pipeline from preprocessing to analysis.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            visualize: Whether to show/save visualizations. Set compute_tsne=True
                       explicitly to calculate t-SNE coordinates.
            extract_keywords: Whether to extract keywords
            analyze_topics: Whether to perform topic analysis
            min_clusters: Minimum clusters required (for tuning)
            time_limit: Tuning time limit in seconds
            random_state: Random state for reproducibility. If None, uses instance random_state
            **kwargs: Additional preprocessing/clustering arguments, including
                      umap_metric, hdbscan_metric, standardize, compute_tsne,
                      tsne_perplexity, tsne_early_exaggeration, tsne_learning_rate,
                      max_pairwise_memory_mb, min_coverage, and max_iterations (tuning only).
            
        Returns:
            Dictionary containing all results
        """
        # Use instance random state if not provided
        rs = random_state if random_state is not None else self.random_state
        
        results = {}
        errors = {}
        
        # 1. Preprocessing
        print("Step 1/5: Preprocessing data...")
        preprocess_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['custom_stopwords', 'keep_words', 'low_idf', 'idf_threshold']}
        self.preprocess_data(**preprocess_kwargs)
        
        # 2. Clustering
        print("Step 2/5: Clustering data...")
        compute_tsne = kwargs.get('compute_tsne', visualize)
        common_options = {
            'metric', 'umap_metric', 'hdbscan_metric', 'standardize',
            'tsne_perplexity', 'tsne_early_exaggeration', 'tsne_learning_rate',
            'max_pairwise_memory_mb', 'min_coverage'
        }
        if tune_hyperparameters:
            clustering_kwargs = {k: v for k, v in kwargs.items() 
                               if k in common_options | {'target_metric', 'tune_tsne',
                                                         'tsne_param_grid', 'max_iterations'}}
            results['clustering'] = self.tune_and_cluster(
                min_clusters=min_clusters,
                time_limit=time_limit,
                random_state=rs,
                compute_tsne=compute_tsne,
                **clustering_kwargs
            )
        else:
            clustering_kwargs = {k: v for k, v in kwargs.items() 
                               if k in common_options | {'n_neighbors', 'min_dist', 'n_components',
                                                         'min_cluster_size', 'cluster_selection_method'}}
            results['clustering'] = self.cluster_data(
                random_state=rs, compute_tsne=compute_tsne, **clustering_kwargs
            )
        
        # 3. Visualization
        if visualize and compute_tsne:
            print("Step 3/5: Visualizing clusters...")
            try:
                viz_kwargs = {k: v for k, v in kwargs.items() 
                            if k in ['show_outliers', 'annotation_column', 'save_path']}
                self.visualize_clusters(**viz_kwargs)
            except Exception as e:
                logger.warning("Visualization failed: %s", e)
                errors['visualization'] = f"{type(e).__name__}: {e}"
        else:
            print("Step 3/5: Skipping visualization...")
        
        # 4. Keyword Extraction
        if extract_keywords:
            print("Step 4/5: Extracting keywords...")
            try:
                keyword_kwargs = {k: v for k, v in kwargs.items() 
                                if k in ['tf_top_n', 'yake_top_n', 'yake_final_n']}
                results['keywords'] = self.extract_keywords(**keyword_kwargs)
            except Exception as e:
                logger.warning("Keyword extraction failed: %s", e)
                errors['keywords'] = f"{type(e).__name__}: {e}"
                results['keywords'] = None
        else:
            print("Step 4/5: Skipping keyword extraction...")
            results['keywords'] = None
        
        # 5. Topic Analysis
        if analyze_topics:
            print("Step 5/5: Analyzing topics...")
            try:
                topic_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['num_topics', 'passes', 'designators',
                                       'designator_profile', 'min_similarity']}
                results['topics'] = self.analyze_topics(random_state=rs, **topic_kwargs)
            except Exception as e:
                logger.warning("Topic analysis failed: %s", e)
                errors['topics'] = f"{type(e).__name__}: {e}"
                results['topics'] = None
        else:
            print("Step 5/5: Skipping topic analysis...")
            results['topics'] = None
        
        results['dataframe'] = self.df
        if errors:
            results['errors'] = errors
        print("\nPipeline complete!")
        
        return results
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary of cluster sizes and statistics.
        
        Returns:
            DataFrame with cluster statistics
        """
        if 'Clusters' not in self.df.columns:
            raise ValueError("No cluster assignments found. Run clustering first.")
        
        summary = self.df.groupby('Clusters').agg({
            'Clusters': 'count'
        }).rename(columns={'Clusters': 'Size'})
        
        summary['Percentage'] = (summary['Size'] / len(self.df) * 100).round(2)
        
        return summary.sort_values('Size', ascending=False)


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    example_data = {
        'Narrative_long': [
            'Pilot reported engine failure during climbout',
            'Cabin crew noticed smoke in the cockpit',
            'ATC communication issues during approach',
            'Bird strike during takeoff roll',
            'Landing gear warning light malfunction',
            'Engine oil pressure dropped suddenly',
            'Radio communication lost momentarily',
            'Hydraulic system failure detected',
            'Smoke detector activated in cargo hold',
            'Navigation system malfunction reported'
        ] * 10  # Repeat to have enough data points
    }
    df = pd.DataFrame(example_data)
    
    # Initialize pipeline with reproducible random state
    pipeline = ClusteringPipeline(df, random_state=42)
    
    # Preprocess data first
    print("=== Preprocessing Data ===")
    pipeline.preprocess_data()
    
    # NEW METHOD: Cluster and extract keywords in one step
    print("\n=== Using Integrated Cluster + Keywords Method ===")
    results = pipeline.cluster_and_extract_keywords(
        min_cluster_size=5,
        tf_top_n=5,
        yake_top_n=10,
        yake_final_n=5
    )
    
    # Access results
    print("\n=== Results ===")
    print("\nCluster Summary:")
    print(results['cluster_summary'])
    
    print("\nKeywords by Cluster:")
    print(results['keywords'])
    
    # You can also still use the original separate methods
    print("\n=== Alternative: Using Separate Methods ===")
    pipeline2 = ClusteringPipeline(df, random_state=42)
    pipeline2.preprocess_data()
    pipeline2.cluster_data(min_cluster_size=5)
    keywords = pipeline2.extract_keywords()
    print(keywords)
    
    # MORE DETAILED EXAMPLE: Using cluster_and_extract_keywords with custom parameters
    print("\n" + "="*60)
    print("=== Detailed Example: Integrated Clustering + Keywords ===")
    print("="*60)
    
    # Create a new pipeline instance
    pipeline3 = ClusteringPipeline(df, random_state=42)
    
    # Step 1: Preprocess with custom stopwords
    print("\n1. Preprocessing with custom settings...")
    pipeline3.preprocess_data(
        custom_stopwords=['pilot', 'crew'],  # Domain-specific stopwords
        low_idf=True,
        idf_threshold=1.4
    )
    
    # Step 2: Cluster and extract keywords in one call
    print("\n2. Running integrated clustering + keyword extraction...")
    integrated_results = pipeline3.cluster_and_extract_keywords(
        # Clustering parameters
        n_neighbors=15,              # UMAP neighborhood size
        min_dist=0.0,                # UMAP minimum distance
        metric='cosine',             # Distance metric
        n_components=5,              # UMAP dimensions
        min_cluster_size=5,          # Minimum points per cluster
        cluster_selection_method='eom',  # HDBSCAN method
        standardize=True,            # Standardize features
        compute_tsne=False,          # Opt in to the expensive t-SNE visualization
        # Keyword extraction parameters
        tf_top_n=5,                  # Top TF-IDF terms per cluster
        yake_top_n=10,               # Initial YAKE keywords
        yake_final_n=5               # Final YAKE keywords to keep
    )
    
    # Step 3: Examine the results
    print("\n3. Results Summary:")
    print("-" * 60)
    
    # Cluster summary
    print("\nCluster Distribution:")
    print(integrated_results['cluster_summary'])
    
    # Keywords for each cluster
    print("\nTop Keywords per Cluster:")
    print(integrated_results['keywords'].head(10))
    
    # Access the updated dataframe
    df_with_clusters = integrated_results['dataframe']
    print(f"\nDataFrame now has {len(df_with_clusters.columns)} columns including 'Clusters'")
    
    # Show sample of clustered data
    print("\nSample of Clustered Data:")
    print(df_with_clusters[['Narrative_long', 'Clusters']].head())
    
    # NEW EXAMPLE: Using cluster_and_analyze_topics for LDA topic modeling
    print("\n" + "="*60)
    print("=== Example: Integrated Clustering + LDA Topics ===")
    print("="*60)
    
    # Create another pipeline instance with explicit random state
    pipeline4 = ClusteringPipeline(df, random_state=42)
    
    # Preprocess
    print("\n1. Preprocessing data...")
    pipeline4.preprocess_data()
    
    # Cluster and analyze topics in one call
    print("\n2. Running integrated clustering + LDA topic analysis...")
    topic_results = pipeline4.cluster_and_analyze_topics(
        # Clustering parameters
        min_cluster_size=5,
        n_neighbors=15,
        # LDA parameters
        num_topics=10,               # Number of topics to extract
        passes=15,                   # LDA training passes
        designators=None            # Optional: custom designator dictionary
    )
    
    # Examine topic results
    print("\n3. Topic Analysis Results:")
    print("-" * 60)
    
    # Show cluster topics
    print("\nTopics by Cluster:")
    for cluster_id, topics_list in topic_results['topics']['cluster_topics'].items():
        print(f"\nCluster {cluster_id}:")
        for topic_words, weight in topics_list[:3]:  # Show top 3 topics
            print(f"  - {topic_words} (weight: {weight:.3f})")
    
    # If designators were matched
    if topic_results['topics']['topic_designators']:
        print("\nTopic-to-Designator Matches:")
        for topic_id, matches in list(topic_results['topics']['topic_designators'].items())[:5]:
            print(f"\nTopic {topic_id}:")
            for designator, similarity in matches[:3]:
                print(f"  - {designator} (similarity: {similarity:.2f})")
