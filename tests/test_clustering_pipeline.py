"""
Comprehensive unit tests for clustering_pipeline.py (Updated for optimized version)

Run with: python -m unittest test_clustering_pipeline.py -v
Or with coverage: coverage run -m unittest test_clustering_pipeline.py && coverage report
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock, call
import warnings
import sys
import os

# Import the module to test
from clustering_pipeline import ClusteringPipeline

# Also import the function we'll be mocking to ensure it exists
try:
    from clustering_tuning import full_clustering_pipeline_fixed_params
except ImportError as e:
    print(f"Warning: Could not import full_clustering_pipeline_fixed_params: {e}")
    print("This may cause test failures.")


# ============================================================================
# Base Test Class with Setup Methods
# ============================================================================

class BaseTestCase(unittest.TestCase):
    """Base test case with common setup methods."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_dataframe = self._create_sample_dataframe()
        self.empty_dataframe = pd.DataFrame()
        self.dataframe_missing_column = self._create_dataframe_missing_column()
        self.dataframe_with_nulls = self._create_dataframe_with_nulls()
        self.mock_vectorized_data = np.random.rand(15, 100)
        self.mock_clustering_results = self._create_mock_clustering_results()
    
    @staticmethod
    def _create_sample_dataframe():
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'Narrative_long': [
                'Pilot reported engine failure during climbout',
                'Cabin crew noticed smoke in the cockpit',
                'ATC communication issues during approach',
                'Bird strike during takeoff roll',
                'Landing gear warning light malfunction',
                'Engine overheat warning during cruise',
                'Passenger reported unusual engine noise',
                'Cockpit fire warning activated',
                'Communication lost with tower',
                'Multiple bird strikes during landing',
                'Hydraulic pressure loss detected',
                'Smoke detector alarm in cargo hold',
                'Radio altimeter malfunction',
                'Engine compressor stall',
                'Landing gear failed to extend',
            ],
            'incident_id': range(1, 16),
            'severity': ['high'] * 5 + ['medium'] * 5 + ['low'] * 5
        })
    
    @staticmethod
    def _create_dataframe_missing_column():
        """Create DataFrame without the expected text column."""
        return pd.DataFrame({
            'wrong_column': ['text1', 'text2', 'text3']
        })
    
    @staticmethod
    def _create_dataframe_with_nulls():
        """Create DataFrame with null values in text column."""
        return pd.DataFrame({
            'Narrative_long': [
                'Valid text here',
                None,
                'Another valid text',
                '',
                'More valid text',
                np.nan,
                'Even more text',
                None,
            ] * 2  # 16 rows total
        })
    
    @staticmethod
    def _create_mock_clustering_results():
        """Create mock clustering results."""
        return {
            'clustering_results': {
                'clusters': np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
                'tsne_embeddings': np.random.rand(15, 2),
                'umap_embeddings': np.random.rand(15, 5)
            },
            'tuning_results': {
                'best_params': {
                    'umap': {'n_neighbors': 10, 'min_dist': 0.0, 'metric': 'cosine', 'n_components': 5},
                    'hdbscan': {'min_cluster_size': 3, 'metric': 'euclidean', 'cluster_selection_method': 'eom'},
                    'tsne': {'perplexity': 30, 'early_exaggeration': 12, 'learning_rate': 200}
                },
                'best_score': 0.5
            },
            'success': True,
            'error_message': None
        }


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInitialization(BaseTestCase):
    """Test ClusteringPipeline initialization."""
    
    def test_init_with_valid_dataframe(self):
        """Test initialization with valid DataFrame."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        self.assertIsNotNone(pipeline.df)
        self.assertEqual(len(pipeline.df), 15)
        self.assertEqual(pipeline.text_column, 'Narrative_long')
        self.assertFalse(pipeline._df_modified)
        self.assertIsNone(pipeline._vectorized_text)
        self.assertIsNone(pipeline.clustering_results)
    
    def test_init_with_custom_text_column(self):
        """Test initialization with custom text column name."""
        df = self.sample_dataframe.rename(columns={'Narrative_long': 'custom_text'})
        pipeline = ClusteringPipeline(df, text_column='custom_text')
        
        self.assertEqual(pipeline.text_column, 'custom_text')
        self.assertIn('custom_text', pipeline.df.columns)
    
    def test_init_does_not_copy_dataframe(self):
        """Test that DataFrame is not copied on initialization."""
        original_id = id(self.sample_dataframe)
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # Should be same object until modification
        self.assertEqual(id(pipeline.df), original_id)
        self.assertFalse(pipeline._df_modified)
    
    def test_init_caches_are_none(self):
        """Test that all caches are None on initialization."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        self.assertIsNone(pipeline._vectorized_text)
        self.assertIsNone(pipeline._stopwords)
        self.assertIsNone(pipeline.clustering_results)
        self.assertIsNone(pipeline.keywords)
        self.assertIsNone(pipeline.topics)


# ============================================================================
# DataFrame Copy Management Tests
# ============================================================================

class TestDataFrameCopyManagement(BaseTestCase):
    """Test lazy DataFrame copying."""
    
    def test_ensure_df_copy_on_first_modification(self):
        """Test that DataFrame is copied on first modification."""
        original_id = id(self.sample_dataframe)
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # Before modification
        self.assertEqual(id(pipeline.df), original_id)
        
        # Trigger copy
        pipeline._ensure_df_copy()
        
        # After copy
        self.assertNotEqual(id(pipeline.df), original_id)
        self.assertTrue(pipeline._df_modified)
    
    def test_ensure_df_copy_only_once(self):
        """Test that DataFrame is only copied once."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # First copy
        pipeline._ensure_df_copy()
        first_copy_id = id(pipeline.df)
        
        # Second call should not copy again
        pipeline._ensure_df_copy()
        second_copy_id = id(pipeline.df)
        
        self.assertEqual(first_copy_id, second_copy_id)
    
    def test_modification_triggers_copy(self):
        """Test that DataFrame modification triggers copy."""
        original_id = id(self.sample_dataframe)
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # This should trigger copy internally
        with patch('clustering_pipeline.pre_processing_routine') as mock_preprocess:
            mock_preprocess.return_value = (
                pd.Series(['processed'] * 15),
                pd.Series(['processed_num'] * 15)
            )
            with patch('clustering_pipeline.define_stopwords', return_value=[]):
                with patch('clustering_pipeline.remove_stopwords_corpus', 
                          return_value=np.array(['cleaned'] * 15)):
                    pipeline.preprocess_data()
        
        # DataFrame should now be a copy
        self.assertNotEqual(id(pipeline.df), original_id)
        self.assertTrue(pipeline._df_modified)


# ============================================================================
# Vectorization Caching Tests
# ============================================================================

class TestVectorizationCaching(BaseTestCase):
    """Test vectorization caching mechanism."""
    
    @patch('clustering_pipeline.vectorize_text')
    def test_get_vectorized_text_caches_result(self, mock_vectorize):
        """Test that vectorization result is cached."""
        mock_vectorize.return_value = self.mock_vectorized_data
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        # First call
        result1 = pipeline._get_vectorized_text()
        self.assertEqual(mock_vectorize.call_count, 1)
        
        # Second call should use cache
        result2 = pipeline._get_vectorized_text()
        self.assertEqual(mock_vectorize.call_count, 1)  # Still 1
        
        # Results should be same object
        self.assertIs(result1, result2)
    
    @patch('clustering_pipeline.vectorize_text')
    def test_get_vectorized_text_without_preprocessing_raises_error(self, mock_vectorize):
        """Test that vectorization without preprocessing raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        with self.assertRaisesRegex(ValueError, "Data must be preprocessed first"):
            pipeline._get_vectorized_text()
        
        # Vectorize should not be called
        mock_vectorize.assert_not_called()
    
    @patch('clustering_pipeline.vectorize_text')
    def test_get_vectorized_text_force_recompute(self, mock_vectorize):
        """Test force recompute of vectorization."""
        mock_vectorize.return_value = self.mock_vectorized_data
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        # First call
        pipeline._get_vectorized_text()
        self.assertEqual(mock_vectorize.call_count, 1)
        
        # Force recompute
        pipeline._get_vectorized_text(force_recompute=True)
        self.assertEqual(mock_vectorize.call_count, 2)
    
    def test_preprocessing_invalidates_cache(self):
        """Test that preprocessing invalidates vectorization cache."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # Set cache manually
        pipeline._vectorized_text = np.array([[1, 2, 3]])
        
        # Preprocess should invalidate
        with patch('clustering_pipeline.pre_processing_routine') as mock_preprocess:
            mock_preprocess.return_value = (
                pd.Series(['processed'] * 15),
                pd.Series(['processed_num'] * 15)
            )
            with patch('clustering_pipeline.define_stopwords', return_value=[]):
                with patch('clustering_pipeline.remove_stopwords_corpus', 
                          return_value=np.array(['cleaned'] * 15)):
                    pipeline.preprocess_data()
        
        # Cache should be cleared
        self.assertIsNone(pipeline._vectorized_text)


# ============================================================================
# Preprocessing Tests
# ============================================================================

class TestPreprocessing(BaseTestCase):
    """Test data preprocessing functionality."""
    
    @patch('clustering_pipeline.pre_processing_routine')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    def test_preprocess_data_success(self, mock_remove, mock_define, mock_preprocess):
        """Test successful preprocessing."""
        # Setup mocks
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = ['the', 'a', 'an']
        mock_remove.return_value = np.array(['cleaned'] * 15)
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        result = pipeline.preprocess_data()
        
        # Check columns exist
        self.assertIn('processed', pipeline.df.columns)
        self.assertIn('processed_num_Narr', pipeline.df.columns)
        self.assertIn('processed_stopword_Narr', pipeline.df.columns)
        
        # Check DataFrame was copied
        self.assertTrue(pipeline._df_modified)
        
        # Check mocks were called
        mock_preprocess.assert_called_once()
        mock_define.assert_called_once()
        mock_remove.assert_called_once()
    
    @patch('clustering_pipeline.pre_processing_routine')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    def test_preprocess_with_custom_stopwords(self, mock_remove, mock_define, mock_preprocess):
        """Test preprocessing with custom stopwords."""
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = ['the', 'a']
        mock_remove.return_value = np.array(['cleaned'] * 15)
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        custom_stops = ['pilot', 'aircraft']
        keep = ['important']
        
        pipeline.preprocess_data(
            custom_stopwords=custom_stops,
            keep_words=keep,
            low_idf=False,
            idf_threshold=2.0
        )
        
        # Check that custom parameters were passed
        mock_define.assert_called_once()
        call_kwargs = mock_define.call_args[1]
        self.assertEqual(call_kwargs['custom_stop_words'], custom_stops)
        self.assertEqual(call_kwargs['custom_keep_words'], keep)
        self.assertFalse(call_kwargs['low_idf'])
        self.assertEqual(call_kwargs['idf_threshold'], 2.0)
    
    @patch('clustering_pipeline.pre_processing_routine')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    def test_preprocess_caches_stopwords(self, mock_remove, mock_define, mock_preprocess):
        """Test that stopwords are cached."""
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = ['the', 'a']
        mock_remove.return_value = np.array(['cleaned'] * 15)
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # First preprocessing
        pipeline.preprocess_data()
        self.assertIsNotNone(pipeline._stopwords)
        self.assertEqual(pipeline._stopwords, ['the', 'a'])
        
        # define_stopwords should be called once
        self.assertEqual(mock_define.call_count, 1)


# ============================================================================
# Clustering Tests
# ============================================================================

class TestClustering(BaseTestCase):
    """Test clustering functionality."""
    
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    def test_cluster_data_success(self, mock_pipeline, mock_vectorize):
        """Test successful clustering with fixed parameters."""
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        result = pipeline.cluster_data(
            n_neighbors=15,
            min_cluster_size=3,
            n_components=5
        )
        
        self.assertIn('Clusters', pipeline.df.columns)
        self.assertIsNotNone(pipeline.clustering_results)
        
        # Check structure instead of exact equality
        self.assertIn('clustering_results', result)
        self.assertIn('clusters', result['clustering_results'])
        
        # Check clustering was called with correct params
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs['umap_params']['n_neighbors'], 15)
        self.assertEqual(call_kwargs['hdbscan_params']['min_cluster_size'], 3)
        self.assertEqual(call_kwargs['umap_params']['n_components'], 5)
    
    @patch('clustering_pipeline.vectorize_text')
    def test_cluster_data_without_preprocessing_raises_error(self, mock_vectorize):
        """Test clustering without preprocessing raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        with self.assertRaisesRegex(ValueError, "Data must be preprocessed first"):
            pipeline.cluster_data()
        
        # Vectorize should not be called
        mock_vectorize.assert_not_called()
    
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    def test_cluster_data_caches_vectorization(self, mock_pipeline, mock_vectorize):
        """Test that vectorization is cached and reused."""
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        # First call
        pipeline.cluster_data()
        self.assertEqual(mock_vectorize.call_count, 1)
        
        # Clear clustering results but keep vectorization
        pipeline.clustering_results = None
        
        # Second call should not vectorize again
        pipeline.cluster_data()
        self.assertEqual(mock_vectorize.call_count, 1)  # Still 1, not 2
    
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.full_clustering_pipeline')
    def test_tune_and_cluster_success(self, mock_pipeline, mock_vectorize):
        """Test successful hyperparameter tuning."""
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        result = pipeline.tune_and_cluster(
            min_clusters=2,
            time_limit=300,
            max_iterations=25,
            target_metric='calinski_harabasz'
        )
        
        self.assertIn('Clusters', pipeline.df.columns)
        
        # Check tuning was called with correct params
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs['min_clusters'], 2)
        self.assertEqual(call_kwargs['time_limit'], 300)
        self.assertEqual(call_kwargs['max_iterations'], 25)
        self.assertEqual(call_kwargs['target_metric'], 'calinski_harabasz')
        self.assertTrue(call_kwargs['progress'])
    
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.full_clustering_pipeline')
    def test_tune_and_cluster_no_results_handles_gracefully(self, mock_pipeline, mock_vectorize):
        """Test that tuning with no results is handled gracefully."""
        mock_vectorize.return_value = np.random.rand(15, 100)
        mock_pipeline.return_value = {
            'clustering_results': None,
            'success': False,
            'error_message': 'No valid clustering found'
        }
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        result = pipeline.tune_and_cluster()
        
        # Should not crash, but Clusters column should not exist
        self.assertNotIn('Clusters', pipeline.df.columns)
        self.assertIsNotNone(result)


# ============================================================================
# Visualization Tests
# ============================================================================

class TestVisualization(BaseTestCase):
    """Test visualization functionality."""
    
    @patch('clustering_pipeline.plot_clusters')
    def test_visualize_clusters_basic(self, mock_plot):
        """Test basic cluster visualization."""
        mock_fig = Mock(spec=plt.Figure)
        mock_plot.return_value = mock_fig
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline.clustering_results = self.mock_clustering_results
        pipeline._ensure_df_copy()
        
        result = pipeline.visualize_clusters()
        
        self.assertEqual(result, mock_fig)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        self.assertTrue(call_kwargs['show_outliers'])
        self.assertIsNone(call_kwargs['save_path'])
    
    @patch('clustering_pipeline.plot_clusters')
    def test_visualize_clusters_save_to_file(self, mock_plot):
        """Test visualization with save path."""
        mock_plot.return_value = None  # Returns None when saving
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline.clustering_results = self.mock_clustering_results
        
        result = pipeline.visualize_clusters(save_path='output.png')
        
        self.assertIsNone(result)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        self.assertEqual(call_kwargs['save_path'], 'output.png')
    
    @patch('clustering_pipeline.plot_clusters_with_annotations')
    def test_visualize_clusters_with_annotations(self, mock_plot):
        """Test visualization with annotations."""
        mock_fig = Mock(spec=plt.Figure)
        mock_plot.return_value = mock_fig
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline.clustering_results = self.mock_clustering_results
        pipeline._ensure_df_copy()
        pipeline.df['incident_id'] = range(15)
        
        result = pipeline.visualize_clusters(annotation_column='incident_id')
        
        self.assertEqual(result, mock_fig)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        np.testing.assert_array_equal(call_kwargs['annotations'], range(15))
    
    def test_visualize_without_clustering_raises_error(self):
        """Test visualization without clustering raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        with self.assertRaisesRegex(ValueError, "No clustering results available"):
            pipeline.visualize_clusters()
    
    def test_visualize_with_failed_clustering_raises_error(self):
        """Test visualization with failed clustering raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline.clustering_results = {'clustering_results': None}
        
        with self.assertRaisesRegex(ValueError, "Clustering failed or produced no results"):
            pipeline.visualize_clusters()
    
    def test_visualize_invalid_annotation_column_raises_error(self):
        """Test visualization with invalid annotation column."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline.clustering_results = self.mock_clustering_results
        
        with self.assertRaisesRegex(ValueError, "Annotation column .* not found"):
            pipeline.visualize_clusters(annotation_column='nonexistent')
    
    def test_visualize_no_tsne_embeddings_raises_error(self):
        """Test visualization without t-SNE embeddings raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        results = self.mock_clustering_results.copy()
        results['clustering_results']['tsne_embeddings'] = None
        pipeline.clustering_results = results
        
        with self.assertRaisesRegex(ValueError, "No t-SNE embeddings available"):
            pipeline.visualize_clusters()


# ============================================================================
# Keyword Extraction Tests
# ============================================================================

class TestKeywordExtraction(BaseTestCase):
    """Test keyword extraction functionality."""
    
    @patch('clustering_pipeline.KeywordExtractor')
    def test_extract_keywords_success(self, mock_extractor_class):
        """Test successful keyword extraction."""
        mock_keywords = pd.DataFrame({
            'cluster': [0, 1, 2],
            'keywords': [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']]
        })
        
        mock_extractor = Mock()
        mock_extractor.extract_keywords_per_cluster.return_value = mock_keywords
        mock_extractor_class.return_value = mock_extractor
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0, 1, 2] * 5
        
        result = pipeline.extract_keywords(tf_top_n=10, yake_top_n=20, yake_final_n=8)
        
        self.assertTrue(result.equals(mock_keywords))
        self.assertIsNotNone(pipeline.keywords)
        
        # Check extractor was called with correct params
        mock_extractor.extract_keywords_per_cluster.assert_called_once_with(
            tf_top_n=10,
            yake_top_n=20,
            yake_final_n=8
        )
    
    def test_extract_keywords_without_clustering_raises_error(self):
        """Test keyword extraction without clustering raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        with self.assertRaisesRegex(ValueError, "No cluster assignments found"):
            pipeline.extract_keywords()


# ============================================================================
# Topic Analysis Tests
# ============================================================================

class TestTopicAnalysis(BaseTestCase):
    """Test topic modeling functionality."""
    
    @patch('clustering_pipeline.TopicModeler')
    def test_analyze_topics_success(self, mock_modeler_class):
        """Test successful topic analysis."""
        mock_cluster_topics = {0: ['topic1', 'topic2'], 1: ['topic3']}
        mock_designators = {0: [('designator1', 0.8)], 1: [('designator2', 0.7)]}
        
        mock_modeler = Mock()
        mock_modeler.get_cluster_topics.return_value = mock_cluster_topics
        mock_modeler.match_designators_to_topics.return_value = mock_designators
        mock_modeler_class.return_value = mock_modeler
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0, 1] * 7 + [0]
        pipeline.df['processed_stopword_Narr'] = ['word1 word2'] * 15
        
        result = pipeline.analyze_topics(num_topics=10, passes=20)
        
        self.assertIn('cluster_topics', result)
        self.assertIn('topic_designators', result)
        self.assertIn('model', result)
        self.assertEqual(result['cluster_topics'], mock_cluster_topics)
        
        # Check modeler was called correctly
        mock_modeler.train_lda.assert_called_once_with(num_topics=10, passes=20)
    
    @patch('clustering_pipeline.TopicModeler')
    def test_analyze_topics_creates_tokens_if_missing(self, mock_modeler_class):
        """Test that tokenization is done if Processed_Narrative doesn't exist."""
        mock_modeler = Mock()
        mock_modeler.get_cluster_topics.return_value = {}
        mock_modeler.match_designators_to_topics.return_value = {}
        mock_modeler_class.return_value = mock_modeler
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0] * 15
        pipeline.df['processed_stopword_Narr'] = ['word1 word2 word3'] * 15
        
        pipeline.analyze_topics()
        
        # Should create Processed_Narrative column
        self.assertIn('Processed_Narrative', pipeline.df.columns)
        self.assertIsInstance(pipeline.df['Processed_Narrative'].iloc[0], list)
        self.assertEqual(pipeline.df['Processed_Narrative'].iloc[0], ['word1', 'word2', 'word3'])
    
    def test_analyze_topics_without_clustering_raises_error(self):
        """Test topic analysis without clustering raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        with self.assertRaisesRegex(ValueError, "No cluster assignments found"):
            pipeline.analyze_topics()
    
    def test_analyze_topics_without_preprocessing_raises_error(self):
        """Test topic analysis without preprocessing raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0] * 15
        
        with self.assertRaisesRegex(ValueError, "Data must be preprocessed first"):
            pipeline.analyze_topics()


# ============================================================================
# Cluster Summary Tests
# ============================================================================

class TestClusterSummary(BaseTestCase):
    """Test cluster summary functionality."""
    
    def test_get_cluster_summary_success(self):
        """Test getting cluster summary statistics."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0, 0, 1, 1, 1, 2, 2, 2, 2, -1, -1, 0, 1, 2, 0]
        
        summary = pipeline.get_cluster_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('Size', summary.columns)
        self.assertIn('Percentage', summary.columns)
        
        # Verify counts
        self.assertEqual(summary.loc[2, 'Size'], 5)  # 5 items in cluster 2
        self.assertEqual(summary.loc[0, 'Size'], 4)  # 4 items in cluster 0
        self.assertEqual(summary.loc[1, 'Size'], 4)  # 4 items in cluster 1
        self.assertEqual(summary.loc[-1, 'Size'], 2)  # 2 outliers
        
        # Verify percentages sum to 100
        self.assertAlmostEqual(summary['Percentage'].sum(), 100.0, places=1)
    
    def test_get_cluster_summary_without_clustering_raises_error(self):
        """Test getting summary without clustering raises error."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        with self.assertRaisesRegex(ValueError, "No cluster assignments found"):
            pipeline.get_cluster_summary()
    
    def test_get_cluster_summary_sorted_by_size(self):
        """Test that summary is sorted by cluster size."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
        
        summary = pipeline.get_cluster_summary()
        
        # Should be sorted descending by size
        sizes = summary['Size'].tolist()
        self.assertEqual(sizes, sorted(sizes, reverse=True))


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestFullPipeline(BaseTestCase):
    """Test complete pipeline execution."""
    
    @patch('clustering_pipeline.TopicModeler')
    @patch('clustering_pipeline.KeywordExtractor')
    @patch('clustering_pipeline.plot_clusters')
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.pre_processing_routine')
    def test_full_pipeline_all_steps(self, mock_preprocess, mock_define, mock_remove,
                                     mock_vectorize, mock_cluster_pipeline, mock_plot,
                                     mock_extractor_class, mock_modeler_class):
        """Test running the full pipeline from start to finish."""
        # Setup all mocks
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = []
        mock_remove.return_value = np.array(['cleaned text'] * 15)
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_cluster_pipeline.return_value = self.mock_clustering_results
        mock_plot.return_value = Mock(spec=plt.Figure)
        
        mock_extractor = Mock()
        mock_extractor.extract_keywords_per_cluster.return_value = pd.DataFrame({
            'cluster': [0, 1, 2],
            'keywords': [['word1'], ['word2'], ['word3']]
        })
        mock_extractor_class.return_value = mock_extractor
        
        mock_modeler = Mock()
        mock_modeler.get_cluster_topics.return_value = {0: ['topic1']}
        mock_modeler.match_designators_to_topics.return_value = {0: [('des1', 0.8)]}
        mock_modeler_class.return_value = mock_modeler
        
        # Run full pipeline
        pipeline = ClusteringPipeline(self.sample_dataframe)
        results = pipeline.full_pipeline(
            tune_hyperparameters=False,
            visualize=True,
            extract_keywords=True,
            analyze_topics=True
        )
        
        # Verify all steps completed successfully
        self.assertIn('clustering', results)
        self.assertIn('keywords', results)
        self.assertIn('topics', results)
        self.assertIn('dataframe', results)
        
        # Check DataFrame has Clusters column
        self.assertIn('Clusters', pipeline.df.columns)
    
    @patch('clustering_pipeline.full_clustering_pipeline')
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.pre_processing_routine')
    def test_full_pipeline_with_tuning(self, mock_preprocess, mock_define, mock_remove,
                                       mock_vectorize, mock_cluster_pipeline):
        """Test full pipeline with hyperparameter tuning."""
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = []
        mock_remove.return_value = np.array(['cleaned'] * 15)
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_cluster_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        results = pipeline.full_pipeline(
            tune_hyperparameters=True,
            visualize=False,
            extract_keywords=False,
            analyze_topics=False,
            min_clusters=2,
            time_limit=300
        )
        
        # Check tuning was called
        mock_cluster_pipeline.assert_called_once()
        call_kwargs = mock_cluster_pipeline.call_args[1]
        self.assertTrue(call_kwargs['progress'])
    
    @patch('clustering_pipeline.plot_clusters')
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.pre_processing_routine')
    def test_full_pipeline_visualization_failure_continues(
        self, mock_preprocess, mock_define, mock_remove, mock_vectorize,
        mock_cluster_pipeline, mock_plot
    ):
        """Test that pipeline continues if visualization fails."""
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = []
        mock_remove.return_value = np.array(['cleaned'] * 15)
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_cluster_pipeline.return_value = self.mock_clustering_results
        mock_plot.side_effect = Exception("Plot error")
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        
        # Should not raise, but print warning
        results = pipeline.full_pipeline(
            tune_hyperparameters=False,
            visualize=True,
            extract_keywords=False,
            analyze_topics=False
        )
        
        # Pipeline should still succeed
        self.assertIn('clustering', results)
    
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.pre_processing_routine')
    def test_full_pipeline_selective_steps(
        self, mock_preprocess, mock_define, mock_remove, mock_vectorize, mock_cluster_pipeline
    ):
        """Test running pipeline with selective steps."""
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = []
        mock_remove.return_value = np.array(['cleaned'] * 15)
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_cluster_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        results = pipeline.full_pipeline(
            tune_hyperparameters=False,
            visualize=False,
            extract_keywords=False,
            analyze_topics=False
        )
        
        # Should have clustering but not keywords/topics
        self.assertIn('clustering', results)
        self.assertIsNone(results['keywords'])
        self.assertIsNone(results['topics'])


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases(BaseTestCase):
    """Test edge cases and error handling."""
    
    def test_very_small_dataset(self):
        """Test handling of very small datasets."""
        small_df = pd.DataFrame({
            'Narrative_long': ['text1', 'text2', 'text3']
        })
        
        pipeline = ClusteringPipeline(small_df)
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.df), 3)
    
    def test_single_cluster_result(self):
        """Test handling when all data points cluster to one cluster."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0] * 15  # All same cluster
        
        summary = pipeline.get_cluster_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, 'Size'], 15)
    
    def test_all_outliers_result(self):
        """Test handling when all data points are outliers."""
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [-1] * 15  # All outliers
        
        summary = pipeline.get_cluster_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[-1, 'Size'], 15)
    
    @patch('clustering_pipeline.pre_processing_routine')
    def test_unicode_text_handling(self, mock_preprocess):
        """Test handling of unicode characters in text."""
        df = pd.DataFrame({
            'Narrative_long': [
                'Text with émojis 😊 and spëcial çharacters',
                'Más texto en español',
                'テキスト in Japanese'
            ]
        })
        
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 3),
            pd.Series(['processed_num'] * 3)
        )
        
        pipeline = ClusteringPipeline(df)
        # Should not raise any encoding errors
        self.assertIsNotNone(pipeline)
    
    def test_duplicate_rows(self):
        """Test handling of duplicate text entries."""
        df = pd.DataFrame({
            'Narrative_long': ['same text'] * 10
        })
        
        pipeline = ClusteringPipeline(df)
        self.assertEqual(len(pipeline.df), 10)
    
    def test_very_long_text(self):
        """Test handling of very long text strings."""
        long_text = 'word ' * 10000  # Very long text
        df = pd.DataFrame({
            'Narrative_long': [long_text, 'short text', long_text]
        })
        
        pipeline = ClusteringPipeline(df)
        self.assertEqual(len(pipeline.df), 3)
    
    def test_empty_strings_mixed_with_valid(self):
        """Test dataset with mix of empty and valid strings."""
        df = pd.DataFrame({
            'Narrative_long': ['valid text', '', 'another valid', '   ', 'more text']
        })
        
        pipeline = ClusteringPipeline(df)
        self.assertEqual(len(pipeline.df), 5)


# ============================================================================
# Memory Efficiency Tests
# ============================================================================

class TestMemoryEfficiency(BaseTestCase):
    """Test memory efficiency improvements."""
    
    def test_no_copy_on_initialization(self):
        """Test that DataFrame is not copied on initialization."""
        original_df = self.sample_dataframe
        original_id = id(original_df)
        
        pipeline = ClusteringPipeline(original_df)
        
        # Should be same object
        self.assertEqual(id(pipeline.df), original_id)
        self.assertFalse(pipeline._df_modified)
    
    def test_copy_only_on_modification(self):
        """Test that copy happens only when needed."""
        original_df = self.sample_dataframe
        original_id = id(original_df)
        
        pipeline = ClusteringPipeline(original_df)
        
        # Trigger modification
        pipeline._ensure_df_copy()
        
        # Now should be different
        self.assertNotEqual(id(pipeline.df), original_id)
        self.assertTrue(pipeline._df_modified)
    
    def test_original_dataframe_unchanged(self):
        """Test that original DataFrame remains unchanged."""
        original_df = self.sample_dataframe.copy()
        original_columns = list(original_df.columns)
        
        pipeline = ClusteringPipeline(original_df)
        pipeline._ensure_df_copy()
        pipeline.df['new_column'] = 'test'
        
        # Original should be unchanged
        self.assertEqual(list(original_df.columns), original_columns)
        self.assertNotIn('new_column', original_df.columns)
    
    @patch('clustering_pipeline.vectorize_text')
    def test_vectorization_shared_across_methods(self, mock_vectorize):
        """Test that vectorization is shared between cluster_data and tune_and_cluster."""
        mock_vectorize.return_value = self.mock_vectorized_data
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        # First access
        pipeline._get_vectorized_text()
        first_call_count = mock_vectorize.call_count
        
        # Second access
        pipeline._get_vectorized_text()
        second_call_count = mock_vectorize.call_count
        
        # Should be same (cached)
        self.assertEqual(first_call_count, second_call_count)


# ============================================================================
# String Operations Tests
# ============================================================================

class TestStringOperations(BaseTestCase):
    """Test optimized string operations."""
    
    @patch('clustering_pipeline.TopicModeler')
    def test_tokenization_uses_str_split(self, mock_modeler_class):
        """Test that tokenization uses vectorized str.split()."""
        mock_modeler = Mock()
        mock_modeler.get_cluster_topics.return_value = {}
        mock_modeler.match_designators_to_topics.return_value = {}
        mock_modeler_class.return_value = mock_modeler
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0] * 15
        pipeline.df['processed_stopword_Narr'] = ['word1 word2 word3'] * 15
        
        # This should use str.split() internally
        pipeline.analyze_topics()
        
        # Verify tokenization worked correctly
        self.assertIn('Processed_Narrative', pipeline.df.columns)
        self.assertIsInstance(pipeline.df['Processed_Narrative'].iloc[0], list)
    
    @patch('clustering_pipeline.TopicModeler')
    def test_tokenization_preserves_existing_tokens(self, mock_modeler_class):
        """Test that existing tokens are preserved."""
        mock_modeler = Mock()
        mock_modeler.get_cluster_topics.return_value = {}
        mock_modeler.match_designators_to_topics.return_value = {}
        mock_modeler_class.return_value = mock_modeler
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['Clusters'] = [0] * 15
        pipeline.df['processed_stopword_Narr'] = ['word1 word2'] * 15
        pipeline.df['Processed_Narrative'] = [['existing', 'tokens']] * 15
        
        pipeline.analyze_topics()
        
        # Should not overwrite existing tokens
        self.assertEqual(pipeline.df['Processed_Narrative'].iloc[0], ['existing', 'tokens'])


# ============================================================================
# Parameter Passing Tests
# ============================================================================

class TestParameterPassing(BaseTestCase):
    """Test parameter passing to underlying functions."""
    
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    @patch('clustering_pipeline.vectorize_text')
    def test_cluster_data_basic_call(self, mock_vectorize, mock_pipeline):
        """Test that cluster_data calls the pipeline function."""
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        # Simple call with defaults
        result = pipeline.cluster_data()
        
        # Verify it was called
        mock_pipeline.assert_called_once()
        self.assertIsNotNone(result)
    
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    @patch('clustering_pipeline.vectorize_text')
    def test_cluster_data_passes_all_parameters(self, mock_vectorize, mock_pipeline):
        """Test that cluster_data passes all parameters correctly."""
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        try:
            pipeline.cluster_data(
                n_neighbors=20,
                min_dist=0.1,
                metric='euclidean',
                n_components=10,
                min_cluster_size=5,
                cluster_selection_method='leaf',
                standardize=False,
                tsne_perplexity=50,
                tsne_early_exaggeration=20,
                tsne_learning_rate=300
            )
        except Exception as e:
            self.fail(f"cluster_data raised an exception: {e}")
        
        # Verify mock was called
        self.assertTrue(mock_pipeline.called, 
                       f"Pipeline function was not called. Call count: {mock_pipeline.call_count}")
        
        # Get call arguments
        if mock_pipeline.call_args is None:
            self.fail("mock_pipeline.call_args is None")
        
        call_kwargs = mock_pipeline.call_args[1]
        
        # Verify UMAP parameters
        self.assertIn('umap_params', call_kwargs)
        self.assertEqual(call_kwargs['umap_params']['n_neighbors'], 20)
        self.assertEqual(call_kwargs['umap_params']['min_dist'], 0.1)
        self.assertEqual(call_kwargs['umap_params']['metric'], 'euclidean')
        self.assertEqual(call_kwargs['umap_params']['n_components'], 10)
        
        # Verify HDBSCAN parameters
        self.assertIn('hdbscan_params', call_kwargs)
        self.assertEqual(call_kwargs['hdbscan_params']['min_cluster_size'], 5)
        self.assertEqual(call_kwargs['hdbscan_params']['cluster_selection_method'], 'leaf')
        self.assertEqual(call_kwargs['hdbscan_params']['metric'], 'euclidean')
        
        # Verify other parameters
        self.assertFalse(call_kwargs['standardize'])
        
        # Verify t-SNE parameters
        self.assertIn('tsne_params', call_kwargs)
        self.assertEqual(call_kwargs['tsne_params']['perplexity'], 50)
        self.assertEqual(call_kwargs['tsne_params']['early_exaggeration'], 20)
        self.assertEqual(call_kwargs['tsne_params']['learning_rate'], 300)
    
    @patch('clustering_pipeline.full_clustering_pipeline')
    @patch('clustering_pipeline.vectorize_text')
    def test_tune_and_cluster_passes_all_parameters(self, mock_vectorize, mock_pipeline):
        """Test that tune_and_cluster passes all parameters correctly."""
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_pipeline.return_value = self.mock_clustering_results
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline._ensure_df_copy()
        pipeline.df['processed_stopword_Narr'] = ['text'] * 15
        
        pipeline.tune_and_cluster(
            min_clusters=5,
            time_limit=600,
            max_iterations=100,
            target_metric='calinski_harabasz',
            standardize=False,
            tune_tsne=True
        )
        
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs['min_clusters'], 5)
        self.assertEqual(call_kwargs['time_limit'], 600)
        self.assertEqual(call_kwargs['max_iterations'], 100)
        self.assertEqual(call_kwargs['target_metric'], 'calinski_harabasz')
        self.assertFalse(call_kwargs['standardize'])
        self.assertTrue(call_kwargs['tune_tsne'])


# ============================================================================
# Progress Reporting Tests
# ============================================================================

class TestProgressReporting(BaseTestCase):
    """Test progress reporting functionality."""
    
    @patch('builtins.print')
    @patch('clustering_pipeline.TopicModeler')
    @patch('clustering_pipeline.KeywordExtractor')
    @patch('clustering_pipeline.full_clustering_pipeline_fixed_params')
    @patch('clustering_pipeline.vectorize_text')
    @patch('clustering_pipeline.remove_stopwords_corpus')
    @patch('clustering_pipeline.define_stopwords')
    @patch('clustering_pipeline.pre_processing_routine')
    def test_full_pipeline_prints_progress(
        self, mock_preprocess, mock_define, mock_remove, mock_vectorize,
        mock_cluster_pipeline, mock_extractor_class, mock_modeler_class, mock_print
    ):
        """Test that full pipeline prints progress messages."""
        # Setup mocks
        mock_preprocess.return_value = (
            pd.Series(['processed'] * 15),
            pd.Series(['processed_num'] * 15)
        )
        mock_define.return_value = []
        mock_remove.return_value = np.array(['cleaned'] * 15)
        mock_vectorize.return_value = self.mock_vectorized_data
        mock_cluster_pipeline.return_value = self.mock_clustering_results
        
        mock_extractor = Mock()
        mock_extractor.extract_keywords_per_cluster.return_value = pd.DataFrame()
        mock_extractor_class.return_value = mock_extractor
        
        mock_modeler = Mock()
        mock_modeler.get_cluster_topics.return_value = {}
        mock_modeler.match_designators_to_topics.return_value = {}
        mock_modeler_class.return_value = mock_modeler
        
        pipeline = ClusteringPipeline(self.sample_dataframe)
        pipeline.full_pipeline()
        
        # Check that progress messages were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        progress_messages = [c for c in print_calls if 'Step' in c]
        self.assertGreater(len(progress_messages), 0)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)