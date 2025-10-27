# test_topic_extraction.py
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import tempfile
import os

# Import the classes to test
from topic_extraction import KeywordExtractor, TopicModeler


class TestKeywordExtractor(unittest.TestCase):
    """Test suite for KeywordExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_df = pd.DataFrame({
            'Clusters': [0, 0, 1, 1, 2],
            'Narratives': [
                'pilot failed checklist',
                'crew missed procedure',
                'weather caused delay',
                'storm diverted flight',
                'engine malfunction occurred'
            ],
            'Narrative_long': [
                'The pilot failed to complete the pre-flight checklist properly',
                'The crew missed an important safety procedure during takeoff',
                'Bad weather conditions caused significant flight delay',
                'Severe storm forced the flight to divert to alternate airport',
                'Critical engine malfunction occurred during cruise phase'
            ]
        })
    
    def test_init_with_valid_dataframe(self):
        """Test initialization with valid DataFrame."""
        extractor = KeywordExtractor(self.valid_df)
        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.cluster_col, 'Clusters')
        self.assertIsNone(extractor.keyframe)
    
    def test_init_with_custom_column_names(self):
        """Test initialization with custom column names."""
        custom_df = self.valid_df.rename(columns={
            'Clusters': 'cluster_id',
            'Narratives': 'text',
            'Narrative_long': 'long_text'
        })
        
        extractor = KeywordExtractor(
            custom_df,
            cluster_col='cluster_id',
            narrative_col='text',
            narrative_long_col='long_text'
        )
        
        self.assertEqual(extractor.cluster_col, 'cluster_id')
        self.assertEqual(extractor.narrative_col, 'text')
    
    def test_init_with_none_dataframe(self):
        """Test that None DataFrame raises ValueError."""
        with self.assertRaises(ValueError) as context:
            KeywordExtractor(None)
        self.assertIn("cannot be None or empty", str(context.exception))
    
    def test_init_with_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            KeywordExtractor(empty_df)
        self.assertIn("cannot be None or empty", str(context.exception))
    
    def test_init_with_missing_columns(self):
        """Test that missing required columns raises ValueError."""
        incomplete_df = pd.DataFrame({
            'Clusters': [0, 1, 2],
            'Narratives': ['text1', 'text2', 'text3']
            # Missing 'Narrative_long'
        })
        
        with self.assertRaises(ValueError) as context:
            KeywordExtractor(incomplete_df)
        self.assertIn("missing required columns", str(context.exception))
        self.assertIn("Narrative_long", str(context.exception))
    
    @patch('topic_extraction.calculate_tf_corpus')
    @patch('topic_extraction.calculate_tfidf_corpus')
    @patch('topic_extraction.calculate_tfdf_corpus')
    @patch('topic_extraction.extract_kw_yake_corpus')
    def test_extract_keywords_per_cluster(self, mock_yake, mock_tfdf, mock_tfidf, mock_tf):
        """Test keyword extraction with mocked utility functions."""
        # Setup mocks to return dictionaries with keywords
        mock_tf.return_value = {'keyword1': 0.5, 'keyword2': 0.3}
        mock_tfidf.return_value = {'keyword3': 0.7, 'keyword4': 0.4}
        mock_tfdf.return_value = {'keyword5': 0.6, 'keyword6': 0.2}
        mock_yake.return_value = {'keyword7': 0.8, 'keyword8': 0.5}
        
        extractor = KeywordExtractor(self.valid_df)
        result = extractor.extract_keywords_per_cluster(tf_top_n=5)
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that all methods were called for each cluster
        num_clusters = self.valid_df['Clusters'].nunique()
        self.assertEqual(mock_tf.call_count, num_clusters)
        self.assertEqual(mock_tfidf.call_count, num_clusters)
        self.assertEqual(mock_tfdf.call_count, num_clusters)
        self.assertEqual(mock_yake.call_count, num_clusters * 2)  # Called for both short and long
        
        # Check DataFrame structure
        self.assertIn('cluster', result.columns)
        self.assertIn('Yake Long', result.columns)
        self.assertIn('TF', result.columns)
        self.assertEqual(len(result), num_clusters)
    
    @patch('topic_extraction.calculate_tf_corpus')
    @patch('topic_extraction.calculate_tfidf_corpus')
    @patch('topic_extraction.calculate_tfdf_corpus')
    @patch('topic_extraction.extract_kw_yake_corpus')
    def test_extract_keywords_with_single_cluster(self, mock_yake, mock_tfdf, mock_tfidf, mock_tf):
        """Test keyword extraction with single cluster."""
        single_cluster_df = pd.DataFrame({
            'Clusters': [0, 0, 0],
            'Narratives': ['text1', 'text2', 'text3'],
            'Narrative_long': ['long text1', 'long text2', 'long text3']
        })
        
        mock_tf.return_value = {'word': 1.0}
        mock_tfidf.return_value = {'word': 1.0}
        mock_tfdf.return_value = {'word': 1.0}
        mock_yake.return_value = {'word': 1.0}
        
        extractor = KeywordExtractor(single_cluster_df)
        result = extractor.extract_keywords_per_cluster()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['cluster'], 0)
    
    @patch('topic_extraction.calculate_tf_corpus')
    @patch('topic_extraction.calculate_tfidf_corpus')
    @patch('topic_extraction.calculate_tfdf_corpus')
    @patch('topic_extraction.extract_kw_yake_corpus')
    def test_save_keywords_success(self, mock_yake, mock_tfdf, mock_tfidf, mock_tf):
        """Test successful saving of keywords to file."""
        mock_tf.return_value = {'word': 1.0}
        mock_tfidf.return_value = {'word': 1.0}
        mock_tfdf.return_value = {'word': 1.0}
        mock_yake.return_value = {'word': 1.0}
        
        extractor = KeywordExtractor(self.valid_df)
        extractor.extract_keywords_per_cluster()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            extractor.save_keywords(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Verify file content
            saved_df = pd.read_csv(temp_path)
            self.assertIn('cluster', saved_df.columns)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_save_keywords_without_extraction(self):
        """Test that saving before extraction raises ValueError."""
        extractor = KeywordExtractor(self.valid_df)
        
        with self.assertRaises(ValueError) as context:
            extractor.save_keywords('test.csv')
        self.assertIn("No keywords extracted yet", str(context.exception))
    
    def test_extract_keywords_with_empty_cluster(self):
        """Test behavior with DataFrame containing no data for a cluster ID."""
        # This is an edge case - normally clusters would be contiguous
        df_with_gap = pd.DataFrame({
            'Clusters': [0, 2],  # Missing cluster 1
            'Narratives': ['text1', 'text2'],
            'Narrative_long': ['long1', 'long2']
        })
        
        with patch('topic_extraction.calculate_tf_corpus') as mock_tf, \
             patch('topic_extraction.calculate_tfidf_corpus') as mock_tfidf, \
             patch('topic_extraction.calculate_tfdf_corpus') as mock_tfdf, \
             patch('topic_extraction.extract_kw_yake_corpus') as mock_yake:
            
            mock_tf.return_value = {'word': 1.0}
            mock_tfidf.return_value = {'word': 1.0}
            mock_tfdf.return_value = {'word': 1.0}
            mock_yake.return_value = {'word': 1.0}
            
            extractor = KeywordExtractor(df_with_gap)
            result = extractor.extract_keywords_per_cluster()
            
            # Should only have 2 rows (for clusters 0 and 2)
            self.assertEqual(len(result), 2)


class TestTopicModeler(unittest.TestCase):
    """Test suite for TopicModeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            ['pilot', 'failed', 'checklist'],
            ['crew', 'missed', 'procedure'],
            ['weather', 'caused', 'delay'],
            ['storm', 'diverted', 'flight'],
            ['engine', 'malfunction', 'occurred']
        ]
        self.sample_clusters = [0, 0, 1, 1, 2]
        
        self.sample_designators = {
            "Test Designator 1": "This is a test designator about procedures",
            "Test Designator 2": "This is about weather and environmental factors"
        }
    
    def test_init_with_valid_inputs(self):
        """Test initialization with valid inputs."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        self.assertIsNotNone(modeler)
        self.assertEqual(len(modeler.texts), len(self.sample_texts))
        self.assertIsNotNone(modeler.designators)
    
    def test_init_with_custom_designators(self):
        """Test initialization with custom designators."""
        modeler = TopicModeler(
            self.sample_texts, 
            self.sample_clusters, 
            designators=self.sample_designators
        )
        self.assertEqual(modeler.designators, self.sample_designators)
    
    def test_init_with_empty_texts(self):
        """Test that empty texts list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            TopicModeler([], [])
        self.assertIn("texts cannot be empty", str(context.exception))
    
    def test_init_with_empty_clusters(self):
        """Test that empty clusters list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            TopicModeler([['word']], [])
        self.assertIn("clusters cannot be empty", str(context.exception))
    
    def test_init_with_mismatched_lengths(self):
        """Test that mismatched texts and clusters lengths raises ValueError."""
        with self.assertRaises(ValueError) as context:
            TopicModeler(self.sample_texts, [0, 1])  # Wrong length
        self.assertIn("Length mismatch", str(context.exception))
    
    def test_device_setup_cpu(self):
        """Test device setup defaults to CPU when GPU not available."""
        with patch('torch.cuda.is_available', return_value=False):
            modeler = TopicModeler(self.sample_texts, self.sample_clusters, use_gpu=True)
            self.assertEqual(modeler.device.type, 'cpu')
    
    def test_device_setup_gpu(self):
        """Test device setup uses GPU when available."""
        with patch('torch.cuda.is_available', return_value=True):
            modeler = TopicModeler(self.sample_texts, self.sample_clusters, use_gpu=True)
            self.assertEqual(modeler.device.type, 'cuda')
    
    def test_device_setup_force_cpu(self):
        """Test forcing CPU even when GPU is available."""
        with patch('torch.cuda.is_available', return_value=True):
            modeler = TopicModeler(self.sample_texts, self.sample_clusters, use_gpu=False)
            self.assertEqual(modeler.device.type, 'cpu')
    
    def test_default_designators(self):
        """Test that default designators are comprehensive."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        designators = modeler.designators
        
        self.assertIsInstance(designators, dict)
        self.assertGreater(len(designators), 5)  # Should have multiple designators
        
        # Check for key designator categories
        designator_names = list(designators.keys())
        has_knowledge = any('knowledge' in name.lower() for name in designator_names)
        has_judgment = any('judgment' in name.lower() or 'decision' in name.lower() for name in designator_names)
        
        self.assertTrue(has_knowledge or has_judgment)
    
    def test_prepare_corpus(self):
        """Test corpus preparation."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler.prepare_corpus()
        
        self.assertIsNotNone(modeler.dictionary)
        self.assertIsNotNone(modeler.corpus)
        self.assertEqual(len(modeler.corpus), len(self.sample_texts))
    
    def test_train_lda(self):
        """Test LDA training."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler.train_lda(num_topics=3, passes=2)
        
        self.assertIsNotNone(modeler.lda_model)
        self.assertEqual(modeler.lda_model.num_topics, 3)
    
    def test_train_lda_auto_prepares_corpus(self):
        """Test that train_lda automatically prepares corpus if needed."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        self.assertIsNone(modeler.corpus)
        
        modeler.train_lda(num_topics=3, passes=2)
        
        self.assertIsNotNone(modeler.corpus)
        self.assertIsNotNone(modeler.lda_model)
    
    def test_get_cluster_topics_without_training(self):
        """Test that get_cluster_topics raises error without training."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        
        with self.assertRaises(ValueError) as context:
            modeler.get_cluster_topics()
        self.assertIn("LDA model not trained yet", str(context.exception))
    
    def test_get_cluster_topics(self):
        """Test getting cluster topics after training."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler.train_lda(num_topics=3, passes=2)
        
        cluster_topics = modeler.get_cluster_topics(top_n=2)
        
        self.assertIsInstance(cluster_topics, dict)
        # Should have topics for each unique cluster
        unique_clusters = set(self.sample_clusters)
        self.assertEqual(len(cluster_topics), len(unique_clusters))
        
        # Each cluster should have at most top_n topics
        for cluster_id, topics in cluster_topics.items():
            self.assertLessEqual(len(topics), 2)
    
    def test_get_cluster_topics_single_cluster(self):
        """Test getting topics for single cluster."""
        single_cluster_texts = [['word1', 'word2'], ['word3', 'word4']]
        single_cluster_ids = [0, 0]
        
        modeler = TopicModeler(single_cluster_texts, single_cluster_ids)
        modeler.train_lda(num_topics=2, passes=2)
        
        cluster_topics = modeler.get_cluster_topics(top_n=2)
        
        self.assertEqual(len(cluster_topics), 1)
        self.assertIn(0, cluster_topics)
    
    def test_get_topic_words_without_training(self):
        """Test that get_topic_words raises error without training."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        
        with self.assertRaises(ValueError) as context:
            modeler.get_topic_words(0)
        self.assertIn("LDA model not trained yet", str(context.exception))
    
    def test_get_topic_words(self):
        """Test getting topic words."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler.train_lda(num_topics=3, passes=2)
        
        words = modeler.get_topic_words(0, num_words=5)
        
        self.assertIsInstance(words, list)
        self.assertEqual(len(words), 5)
        self.assertTrue(all(isinstance(word, str) for word in words))
    
    @patch('topic_extraction.BertTokenizer')
    @patch('topic_extraction.BertModel')
    def test_load_bert(self, mock_bert_model, mock_bert_tokenizer):
        """Test BERT loading."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_bert_model.from_pretrained.return_value = mock_model_instance
        
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler.load_bert()
        
        self.assertIsNotNone(modeler.tokenizer)
        self.assertIsNotNone(modeler.bert_model)
        mock_bert_tokenizer.from_pretrained.assert_called_once()
        mock_bert_model.from_pretrained.assert_called_once()
        mock_model_instance.to.assert_called_once()
        mock_model_instance.eval.assert_called_once()
    
    @patch('topic_extraction.BertTokenizer')
    @patch('topic_extraction.BertModel')
    def test_get_bert_embedding(self, mock_bert_model_class, mock_bert_tokenizer_class):
        """Test BERT embedding generation."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_bert_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_bert_model_class.from_pretrained.return_value = mock_model
        
        # Mock tokenizer output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 768)
        mock_model.return_value = mock_outputs
        
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        embedding = modeler.get_bert_embedding("test text")
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[1], 768)  # BERT base hidden size
    
    @patch('topic_extraction.BertTokenizer')
    @patch('topic_extraction.BertModel')
    def test_get_bert_embedding_caching(self, mock_bert_model_class, mock_bert_tokenizer_class):
        """Test that BERT embedding caching works."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_bert_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_bert_model_class.from_pretrained.return_value = mock_model
        
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 768)
        mock_model.return_value = mock_outputs
        
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        
        # First call should compute embedding
        embedding1 = modeler.get_bert_embedding("test text", use_cache=True)
        call_count_1 = mock_model.call_count
        
        # Second call should use cache
        embedding2 = modeler.get_bert_embedding("test text", use_cache=True)
        call_count_2 = mock_model.call_count
        
        # Model should not be called again
        self.assertEqual(call_count_1, call_count_2)
        np.testing.assert_array_equal(embedding1, embedding2)
    
    @patch('topic_extraction.BertTokenizer')
    @patch('topic_extraction.BertModel')
    def test_get_bert_embedding_no_cache(self, mock_bert_model_class, mock_bert_tokenizer_class):
        """Test BERT embedding without caching."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_bert_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_bert_model_class.from_pretrained.return_value = mock_model
        
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 768)
        mock_model.return_value = mock_outputs
        
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        
        # Call twice without caching
        modeler.get_bert_embedding("test text", use_cache=False)
        call_count_1 = mock_model.call_count
        
        modeler.get_bert_embedding("test text", use_cache=False)
        call_count_2 = mock_model.call_count
        
        # Model should be called twice
        self.assertEqual(call_count_2, call_count_1 + 1)
    
    def test_match_designators_without_training(self):
        """Test that match_designators raises error without training."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        
        with self.assertRaises(ValueError) as context:
            modeler.match_designators_to_topics()
        self.assertIn("LDA model not trained yet", str(context.exception))
    
    @patch.object(TopicModeler, 'get_bert_embedding')
    def test_match_designators_to_topics(self, mock_get_embedding):
        """Test matching designators to topics."""
        # Setup mock to return consistent embeddings
        mock_get_embedding.return_value = np.random.randn(1, 768)
        
        modeler = TopicModeler(
            self.sample_texts, 
            self.sample_clusters,
            designators=self.sample_designators
        )
        modeler.train_lda(num_topics=2, passes=2)
        
        matches = modeler.match_designators_to_topics(num_words=5, top_k=2)
        
        self.assertIsInstance(matches, dict)
        self.assertEqual(len(matches), 2)  # One entry per topic
        
        # Each topic should have top_k designators
        for topic_id, designator_list in matches.items():
            self.assertLessEqual(len(designator_list), 2)
            # Each entry should be (designator_name, similarity_score)
            for designator, score in designator_list:
                self.assertIn(designator, self.sample_designators)
                self.assertIsInstance(score, (float, np.floating))
    
    def test_clear_cache(self):
        """Test clearing the embedding cache."""
        modeler = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler.embedding_cache['test'] = np.array([1, 2, 3])
        
        self.assertEqual(len(modeler.embedding_cache), 1)
        
        modeler.clear_cache()
        
        self.assertEqual(len(modeler.embedding_cache), 0)
    
    def test_reproducibility_with_random_state(self):
        """Test that LDA training is reproducible with same random state."""
        modeler1 = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler1.train_lda(num_topics=3, passes=5, random_state=42)
        
        modeler2 = TopicModeler(self.sample_texts, self.sample_clusters)
        modeler2.train_lda(num_topics=3, passes=5, random_state=42)
        
        # Get topics for comparison
        topics1 = [modeler1.lda_model.show_topic(i, 5) for i in range(3)]
        topics2 = [modeler2.lda_model.show_topic(i, 5) for i in range(3)]
        
        # Topics should be identical with same random state
        self.assertEqual(topics1, topics2)
    
    def test_large_cluster_handling(self):
        """Test handling of large number of clusters."""
        # Create data with many clusters
        large_texts = [['word' + str(i)] for i in range(100)]
        large_clusters = list(range(100))  # 100 different clusters
        
        modeler = TopicModeler(large_texts, large_clusters)
        modeler.train_lda(num_topics=10, passes=2)
        
        cluster_topics = modeler.get_cluster_topics(top_n=3)
        
        self.assertEqual(len(cluster_topics), 100)
    
    def test_single_word_documents(self):
        """Test handling of single-word documents."""
        single_word_texts = [['word1'], ['word2'], ['word3']]
        single_word_clusters = [0, 1, 2]
        
        modeler = TopicModeler(single_word_texts, single_word_clusters)
        modeler.train_lda(num_topics=2, passes=2)
        
        self.assertIsNotNone(modeler.lda_model)
    
    def test_very_long_documents(self):
        """Test handling of very long documents."""
        long_texts = [[f'word{i}' for i in range(1000)] for _ in range(3)]
        long_clusters = [0, 1, 2]
        
        modeler = TopicModeler(long_texts, long_clusters)
        modeler.train_lda(num_topics=2, passes=2)
        
        self.assertIsNotNone(modeler.lda_model)


class TestIntegration(unittest.TestCase):
    """Integration tests combining KeywordExtractor and TopicModeler."""
    
    @patch('topic_extraction.calculate_tf_corpus')
    @patch('topic_extraction.calculate_tfidf_corpus')
    @patch('topic_extraction.calculate_tfdf_corpus')
    @patch('topic_extraction.extract_kw_yake_corpus')
    def test_end_to_end_workflow(self, mock_yake, mock_tfdf, mock_tfidf, mock_tf):
        """Test complete workflow from DataFrame to topic modeling."""
        # Setup mocks
        mock_tf.return_value = {'keyword': 1.0}
        mock_tfidf.return_value = {'keyword': 1.0}
        mock_tfdf.return_value = {'keyword': 1.0}
        mock_yake.return_value = {'keyword': 1.0}
        
        # Create test data
        df = pd.DataFrame({
            'Clusters': [0, 0, 1, 1],
            'Narratives': ['text1', 'text2', 'text3', 'text4'],
            'Narrative_long': ['long1', 'long2', 'long3', 'long4']
        })
        
        # Extract keywords
        extractor = KeywordExtractor(df)
        keywords = extractor.extract_keywords_per_cluster()
        
        self.assertIsNotNone(keywords)
        
        # Prepare texts for topic modeling
        texts = [['word1', 'word2'], ['word3', 'word4'], ['word5'], ['word6']]
        clusters = [0, 0, 1, 1]
        
        # Train topic model
        modeler = TopicModeler(texts, clusters)
        modeler.train_lda(num_topics=2, passes=2)
        
        cluster_topics = modeler.get_cluster_topics(top_n=1)
        
        self.assertIsNotNone(cluster_topics)
        self.assertEqual(len(cluster_topics), 2)


if __name__ == '__main__':
    unittest.main()