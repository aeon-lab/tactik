# test_preprocessing.py
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import logging
import sys

# Import functions to test
from preprocessing import (
    pre_processing_routine,
    remove_stopwords_corpus,
    define_stopwords,
    load_spacy_model,
    clear_spacy_cache,
    process_with_spacy,
    expand_contractions,
    remove_special_characters,
    normalize_whitespace,
    clean_and_tokenize_pipeline,
    batch_preprocess
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_sample_dataframe(text_column='Narrative_long'):
    """Create a sample DataFrame for testing with configurable column name."""
    return pd.DataFrame({
        text_column: [
            'The pilot reported engine failure during takeoff.',
            'Aircraft experienced turbulence at 30000 feet.',
            'Captain noticed unusual vibrations in flight.',
            None,
            ''
        ]
    })


def create_mock_spacy_model():
    """Create a mock spaCy model."""
    mock_nlp = Mock()
    mock_nlp.pipe_names = ['tok2vec', 'tagger', 'parser', 'ner', 'lemmatizer']
    mock_nlp.disable_pipes = Mock()
    
    def mock_call(text):
        """Mock spaCy processing."""
        mock_doc = Mock()
        # Create mock tokens
        words = text.split() if text else []
        mock_tokens = []
        for word in words:
            token = Mock()
            token.text = word
            token.lemma_ = word.lower()
            token.pos_ = 'NOUN' if len(word) > 3 else 'DET'
            token.is_stop = word.lower() in ['the', 'a', 'an', 'in', 'at']
            token.is_punct = word in ['.', ',', '!', '?']
            mock_tokens.append(token)
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        return mock_doc
    
    mock_nlp.side_effect = mock_call
    mock_nlp.pipe = Mock(side_effect=lambda texts, **kwargs: [mock_call(t) for t in texts])
    return mock_nlp


# ============================================================================
# Tests for expand_contractions
# ============================================================================

class TestExpandContractions(unittest.TestCase):
    
    def test_basic_contractions(self):
        """Test basic contraction expansion."""
        self.assertEqual(expand_contractions("don't worry"), "do not worry")
        self.assertEqual(expand_contractions("can't wait"), "cannot wait")
        self.assertEqual(expand_contractions("it's working"), "it is working")
    
    def test_multiple_contractions(self):
        """Test multiple contractions in one string."""
        text = "I don't think it's working and we can't fix it"
        result = expand_contractions(text)
        self.assertIn("do not", result)
        self.assertIn("it is", result)
        self.assertIn("cannot", result)
    
    def test_word_boundaries(self):
        """Test that word boundaries are respected."""
        # "don't" should not match in "donuts"
        self.assertEqual(expand_contractions("donuts"), "donuts")
        self.assertEqual(expand_contractions("don't"), "do not")
    
    def test_none_input(self):
        """Test None input."""
        self.assertEqual(expand_contractions(None), "")
    
    def test_nan_input(self):
        """Test NaN input."""
        self.assertEqual(expand_contractions(np.nan), "")
    
    def test_empty_string(self):
        """Test empty string."""
        self.assertEqual(expand_contractions(""), "")
    
    def test_non_string_input(self):
        """Test non-string input conversion."""
        self.assertEqual(expand_contractions(123), "123")
    
    def test_no_contractions(self):
        """Test text without contractions."""
        text = "hello world"
        self.assertEqual(expand_contractions(text), text)
    
    def test_caching(self):
        """Test that contractions dict is cached."""
        expand_contractions("don't")
        self.assertTrue(hasattr(expand_contractions, '_contractions_dict'))
        # Second call should use cached dict
        expand_contractions("can't")


# ============================================================================
# Tests for normalize_whitespace
# ============================================================================

class TestNormalizeWhitespace(unittest.TestCase):
    
    def test_multiple_spaces(self):
        """Test normalization of multiple spaces."""
        self.assertEqual(normalize_whitespace("hello    world"), "hello world")
    
    def test_tabs_and_newlines(self):
        """Test normalization of tabs and newlines."""
        self.assertEqual(normalize_whitespace("hello\t\nworld"), "hello world")
    
    def test_leading_trailing_whitespace(self):
        """Test removal of leading/trailing whitespace."""
        self.assertEqual(normalize_whitespace("  hello world  "), "hello world")
    
    def test_mixed_whitespace(self):
        """Test mixed whitespace characters."""
        text = "  hello  \t\n  world  \r\n  "
        self.assertEqual(normalize_whitespace(text), "hello world")
    
    def test_none_input(self):
        """Test None input."""
        self.assertEqual(normalize_whitespace(None), "")
    
    def test_nan_input(self):
        """Test NaN input."""
        self.assertEqual(normalize_whitespace(np.nan), "")
    
    def test_empty_string(self):
        """Test empty string."""
        self.assertEqual(normalize_whitespace(""), "")
    
    def test_non_string_input(self):
        """Test non-string input."""
        self.assertEqual(normalize_whitespace(123), "123")


# ============================================================================
# Tests for remove_special_characters
# ============================================================================

class TestRemoveSpecialCharacters(unittest.TestCase):
    
    def test_basic_removal(self):
        """Test basic special character removal."""
        result = remove_special_characters("hello@world#123", keep_chars="")
        # Should keep alphanumeric and spaces
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("123", result)
    
    def test_keep_punctuation(self):
        """Test keeping specified punctuation."""
        result = remove_special_characters("hello, world!", keep_chars=",!")
        self.assertIn(",", result)
        self.assertIn("!", result)
    
    def test_unicode_preservation(self):
        """Test Unicode character preservation."""
        result = remove_special_characters("café", preserve_unicode=True)
        self.assertIn("café", result)
    
    def test_unicode_removal(self):
        """Test Unicode character removal when disabled."""
        result = remove_special_characters("café", preserve_unicode=False)
        self.assertNotIn("é", result)
        self.assertEqual(result, "caf")
    
    def test_underscore_removal(self):
        """Test underscore removal with Unicode."""
        result = remove_special_characters("hello_world", preserve_unicode=True)
        self.assertNotIn("_", result)
    
    def test_underscore_keeping(self):
        """Test keeping underscore when specified."""
        result = remove_special_characters("hello_world", keep_chars="_", preserve_unicode=True)
        self.assertIn("_", result)
    
    def test_none_input(self):
        """Test None input."""
        self.assertEqual(remove_special_characters(None), "")
    
    def test_nan_input(self):
        """Test NaN input."""
        self.assertEqual(remove_special_characters(np.nan), "")
    
    def test_empty_string(self):
        """Test empty string."""
        self.assertEqual(remove_special_characters(""), "")
    
    def test_pattern_caching(self):
        """Test that patterns are cached."""
        remove_special_characters("test", keep_chars=".,")
        self.assertTrue(hasattr(remove_special_characters, '_pattern_cache'))
        self.assertGreater(len(remove_special_characters._pattern_cache), 0)


# ============================================================================
# Tests for remove_stopwords_corpus
# ============================================================================

class TestRemoveStopwordsCorpus(unittest.TestCase):
    
    @patch('preprocessing.remove_words_dict')
    def test_basic_stopword_removal(self, mock_remove_words):
        """Test basic stopword removal."""
        mock_remove_words.side_effect = lambda text, words: ' '.join(
            [w for w in text.split() if w not in words]
        )
        
        corpus = ["the quick brown fox", "a fast red dog"]
        stopwords = {"the", "a"}
        result = remove_stopwords_corpus(corpus, stopwords)
        self.assertNotIn("the", result[0])
        self.assertIn("quick", result[0])
    
    def test_empty_stopwords(self):
        """Test with empty stopwords raises error."""
        corpus = ["test"]
        with self.assertRaises(ValueError):
            remove_stopwords_corpus(corpus, [])
    
    def test_none_stopwords(self):
        """Test with None stopwords raises error."""
        corpus = ["test"]
        with self.assertRaises(ValueError):
            remove_stopwords_corpus(corpus, None)
    
    @patch('preprocessing.remove_words_dict')
    def test_corpus_with_none_values(self, mock_remove_words):
        """Test corpus with None values."""
        mock_remove_words.side_effect = lambda text, words: text
        
        corpus = ["hello world", None, "test"]
        stopwords = {"world"}
        result = remove_stopwords_corpus(corpus, stopwords)
        self.assertEqual(result[1], "")
    
    def test_invalid_corpus_type(self):
        """Test invalid corpus type."""
        with self.assertRaises(TypeError):
            remove_stopwords_corpus("not a list", {"the"})
    
    def test_corpus_is_string(self):
        """Test that string corpus raises error."""
        with self.assertRaises(TypeError):
            remove_stopwords_corpus("string", {"the"})


# ============================================================================
# Tests for define_stopwords
# ============================================================================

class TestDefineStopwords(unittest.TestCase):
    
    @patch('preprocessing.calculate_idf')
    def test_default_stopwords(self, mock_idf):
        """Test default stopword generation."""
        mock_idf.return_value = [(0.5, 'the'), (0.8, 'aircraft'), (2.0, 'important')]
        
        df = create_sample_dataframe()
        stopwords = define_stopwords(df)
        self.assertIsInstance(stopwords, list)
        self.assertGreater(len(stopwords), 0)
    
    @patch('preprocessing.calculate_idf')
    def test_custom_stopwords(self, mock_idf):
        """Test adding custom stopwords."""
        mock_idf.return_value = []
        
        df = create_sample_dataframe()
        custom = ['test', 'word']
        stopwords = define_stopwords(df, custom_stop_words=custom)
        self.assertIn('test', stopwords)
        self.assertIn('word', stopwords)
    
    @patch('preprocessing.calculate_idf')
    def test_custom_keep_words(self, mock_idf):
        """Test excluding words with custom_keep_words."""
        mock_idf.return_value = []
        
        df = create_sample_dataframe()
        keep = ['pilot']
        stopwords = define_stopwords(
            df,
            custom_stop_words=['pilot'],
            custom_keep_words=keep
        )
        self.assertNotIn('pilot', stopwords)
    
    @patch('preprocessing.calculate_idf')
    def test_low_idf_disabled(self, mock_idf):
        """Test with low_idf disabled."""
        df = create_sample_dataframe()
        stopwords = define_stopwords(df, low_idf=False)
        # Should not call calculate_idf
        mock_idf.assert_not_called()
        self.assertIsInstance(stopwords, list)
    
    @patch('preprocessing.calculate_idf')
    def test_none_custom_stopwords(self, mock_idf):
        """Test None for custom_stop_words uses defaults."""
        mock_idf.return_value = []
        
        df = create_sample_dataframe()
        stopwords = define_stopwords(df, custom_stop_words=None)
        # Should include default aviation terms
        self.assertTrue('pilot' in stopwords or 'aircraft' in stopwords)
    
    def test_empty_dataframe_with_low_idf(self):
        """Test empty dataframe raises error when low_idf=True."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            define_stopwords(empty_df, low_idf=True)
    
    def test_invalid_custom_stopwords_type(self):
        """Test invalid custom_stop_words type."""
        df = create_sample_dataframe()
        with self.assertRaises(TypeError):
            define_stopwords(df, custom_stop_words="string")
    
    @patch('preprocessing.calculate_idf')
    def test_custom_text_column(self, mock_idf):
        """Test with custom text column."""
        mock_idf.return_value = [(0.5, 'the')]
        
        df = create_sample_dataframe('custom_column')
        stopwords = define_stopwords(df, text_column='custom_column', low_idf=True)
        self.assertIsInstance(stopwords, list)
        # Verify calculate_idf was called with correct column parameter
        mock_idf.assert_called_once()
        call_kwargs = mock_idf.call_args[1]
        self.assertEqual(call_kwargs.get('text_column'), 'custom_column')
    
    def test_missing_text_column(self):
        """Test with missing text column raises error."""
        df = create_sample_dataframe()
        with self.assertRaises(ValueError):
            define_stopwords(df, text_column='nonexistent_column', low_idf=True)


# ============================================================================
# Tests for load_spacy_model
# ============================================================================

class TestLoadSpacyModel(unittest.TestCase):
    
    @patch('preprocessing.spacy.load')
    def test_load_default_model(self, mock_spacy_load):
        """Test loading default model."""
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['ner', 'tagger', 'parser']
        mock_nlp.disable_pipes = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        result = load_spacy_model(use_cache=False)
        mock_spacy_load.assert_called_once_with('en_core_web_lg')
        self.assertEqual(result, mock_nlp)
    
    @patch('preprocessing.spacy.load')
    def test_disable_pipes(self, mock_spacy_load):
        """Test disabling specific pipes."""
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['ner', 'tagger', 'parser']
        mock_nlp.disable_pipes = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        load_spacy_model(disable_pipes=['ner', 'tagger'], use_cache=False)
        mock_nlp.disable_pipes.assert_called_once_with('ner', 'tagger')
    
    @patch('preprocessing.spacy.load')
    def test_invalid_pipe_names(self, mock_spacy_load):
        """Test invalid pipe names raise error."""
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['ner', 'tagger']
        mock_spacy_load.return_value = mock_nlp
        
        with self.assertRaises(ValueError):
            load_spacy_model(disable_pipes=['invalid_pipe'], use_cache=False)
    
    @patch('preprocessing.spacy.load')
    def test_model_caching(self, mock_spacy_load):
        """Test model caching."""
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['ner']
        mock_nlp.disable_pipes = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        # Clear cache first
        clear_spacy_cache()
        
        # First call
        result1 = load_spacy_model(disable_pipes=['ner'], use_cache=True)
        # Second call should use cache
        result2 = load_spacy_model(disable_pipes=['ner'], use_cache=True)
        
        # Should only load once
        self.assertEqual(mock_spacy_load.call_count, 1)
        self.assertEqual(result1, result2)
    
    @patch('preprocessing.spacy.load')
    def test_model_not_installed(self, mock_spacy_load):
        """Test error when model not installed."""
        mock_spacy_load.side_effect = OSError("Model not found")
        
        with self.assertRaises(OSError):
            load_spacy_model(use_cache=False)
    
    def test_invalid_disable_pipes_type(self):
        """Test invalid disable_pipes type."""
        with self.assertRaises(ValueError):
            load_spacy_model(disable_pipes="string", use_cache=False)


# ============================================================================
# Tests for clear_spacy_cache
# ============================================================================

class TestClearSpacyCache(unittest.TestCase):
    
    @patch('preprocessing.spacy.load')
    def test_clear_cache(self, mock_spacy_load):
        """Test clearing the cache."""
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['ner']
        mock_nlp.disable_pipes = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        # Load a model to cache it
        load_spacy_model(disable_pipes=['ner'], use_cache=True)
        
        # Clear cache
        count = clear_spacy_cache()
        self.assertGreaterEqual(count, 0)


# ============================================================================
# Tests for clean_and_tokenize_pipeline
# ============================================================================

class TestCleanAndTokenizePipeline(unittest.TestCase):
    
    def test_basic_pipeline(self):
        """Test basic pipeline execution."""
        mock_nlp = create_mock_spacy_model()
        text = "I don't like flying!"
        result = clean_and_tokenize_pipeline(text, mock_nlp)
        self.assertIsInstance(result, str)
    
    def test_none_input(self):
        """Test None input."""
        mock_nlp = create_mock_spacy_model()
        result = clean_and_tokenize_pipeline(None, mock_nlp)
        self.assertEqual(result, "")
    
    def test_nan_input(self):
        """Test NaN input."""
        mock_nlp = create_mock_spacy_model()
        result = clean_and_tokenize_pipeline(np.nan, mock_nlp)
        self.assertEqual(result, "")
    
    def test_empty_string(self):
        """Test empty string."""
        mock_nlp = create_mock_spacy_model()
        result = clean_and_tokenize_pipeline("", mock_nlp)
        self.assertEqual(result, "")
    
    def test_none_nlp_model(self):
        """Test None nlp model raises error."""
        with self.assertRaises(ValueError):
            clean_and_tokenize_pipeline("test", None)
    
    def test_custom_keep_pos(self):
        """Test custom POS tags."""
        mock_nlp = create_mock_spacy_model()
        text = "test text"
        result = clean_and_tokenize_pipeline(
            text, mock_nlp, keep_pos=['NOUN']
        )
        self.assertIsInstance(result, str)
    
    def test_empty_keep_pos(self):
        """Test empty keep_pos."""
        mock_nlp = create_mock_spacy_model()
        result = clean_and_tokenize_pipeline(
            "test", mock_nlp, keep_pos=[]
        )
        self.assertEqual(result, "")
    
    def test_invalid_keep_pos_type(self):
        """Test invalid keep_pos type."""
        mock_nlp = create_mock_spacy_model()
        with self.assertRaises(TypeError):
            clean_and_tokenize_pipeline(
                "test", mock_nlp, keep_pos="string"
            )


# ============================================================================
# Tests for process_with_spacy
# ============================================================================

class TestProcessWithSpacy(unittest.TestCase):
    
    @patch('preprocessing.load_spacy_model')
    def test_basic_processing(self, mock_load):
        """Test basic spaCy processing."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        
        df = create_sample_dataframe()
        # Add the default column expected by process_with_spacy
        df['processed_stopword_Narr'] = df['Narrative_long']
        result = process_with_spacy(df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))
    
    @patch('preprocessing.load_spacy_model')
    def test_with_tokenizer_function(self, mock_load):
        """Test with custom tokenizer function."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        
        def tokenizer(doc):
            return [token.text for token in doc]
        
        df = create_sample_dataframe()
        df['processed_stopword_Narr'] = df['Narrative_long']
        result = process_with_spacy(
            df,
            spacy_tokenizer=tokenizer
        )
        self.assertIsInstance(result, pd.Series)
    
    def test_invalid_dataframe(self):
        """Test invalid DataFrame."""
        with self.assertRaises(ValueError):
            process_with_spacy("not a dataframe")
    
    @patch('preprocessing.load_spacy_model')
    def test_missing_column(self, mock_load):
        """Test missing column."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        df = pd.DataFrame({'wrong_column': ['test']})
        
        with self.assertRaises(ValueError):
            process_with_spacy(df)
    
    @patch('preprocessing.load_spacy_model')
    def test_empty_dataframe(self, mock_load):
        """Test empty DataFrame."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        df = pd.DataFrame({'processed_stopword_Narr': []})
        
        result = process_with_spacy(df)
        self.assertEqual(len(result), 0)
    
    def test_invalid_tokenizer_type(self):
        """Test invalid tokenizer type."""
        df = create_sample_dataframe()
        df['processed_stopword_Narr'] = df['Narrative_long']
        with self.assertRaises(TypeError):
            process_with_spacy(df, spacy_tokenizer="not callable")
    
    @patch('preprocessing.load_spacy_model')
    def test_custom_text_column(self, mock_load):
        """Test with custom text column."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        
        df = create_sample_dataframe('custom_text')
        result = process_with_spacy(df, text_column='custom_text')
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))


# ============================================================================
# Tests for batch_preprocess
# ============================================================================

class TestBatchPreprocess(unittest.TestCase):
    
    @patch('preprocessing.load_spacy_model')
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    @patch('preprocessing.remove_words_dict')
    @patch('preprocessing.calculate_idf')
    def test_basic_batch_processing(self, mock_idf, mock_remove_words, 
                                    mock_remove_num, mock_punctuation, 
                                    mock_preprocess, mock_load):
        """Test basic batch preprocessing."""
        # Setup mocks
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        mock_remove_words.side_effect = lambda text, words: text
        mock_idf.return_value = [(0.5, 'the'), (2.0, 'important')]
        
        df = create_sample_dataframe()
        result = batch_preprocess(df, copy_dataframe=True)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Final_Processed', result.columns)
        self.assertIn('processed', result.columns)
        self.assertIn('processed_num_Narr', result.columns)
        self.assertIn('processed_stopword_Narr', result.columns)
    
    @patch('preprocessing.load_spacy_model')
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    @patch('preprocessing.remove_words_dict')
    @patch('preprocessing.calculate_idf')
    def test_copy_dataframe_true(self, mock_idf, mock_remove_words,
                                 mock_remove_num, mock_punctuation,
                                 mock_preprocess, mock_load):
        """Test that copy_dataframe=True doesn't modify original."""
        # Setup mocks
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        mock_remove_words.side_effect = lambda text, words: text
        mock_idf.return_value = [(0.5, 'the')]
        
        df = create_sample_dataframe()
        original_cols = list(df.columns)
        
        result = batch_preprocess(df, copy_dataframe=True)
        
        # Original should be unchanged
        self.assertEqual(list(df.columns), original_cols)
        # Result should have new columns
        self.assertGreater(len(result.columns), len(original_cols))
    
    @patch('preprocessing.load_spacy_model')
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    @patch('preprocessing.remove_words_dict')
    @patch('preprocessing.calculate_idf')
    def test_copy_dataframe_false(self, mock_idf, mock_remove_words,
                                  mock_remove_num, mock_punctuation,
                                  mock_preprocess, mock_load):
        """Test that copy_dataframe=False modifies original."""
        # Setup mocks
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        mock_remove_words.side_effect = lambda text, words: text
        mock_idf.return_value = [(0.5, 'the')]
        
        df = create_sample_dataframe()
        original_cols = list(df.columns)
        
        result = batch_preprocess(df, copy_dataframe=False)
        
        # Original should be modified
        self.assertNotEqual(list(df.columns), original_cols)
        self.assertIs(result, df)
    
    def test_invalid_dataframe(self):
        """Test invalid DataFrame."""
        with self.assertRaises(ValueError):
            batch_preprocess("not a dataframe")
    
    @patch('preprocessing.load_spacy_model')
    def test_missing_text_column(self, mock_load):
        """Test missing text column."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        df = pd.DataFrame({'wrong_column': ['test']})
        
        with self.assertRaises(ValueError):
            batch_preprocess(df)
    
    @patch('preprocessing.load_spacy_model')
    def test_empty_dataframe(self, mock_load):
        """Test empty DataFrame."""
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        df = pd.DataFrame({'Narrative_long': []})
        
        result = batch_preprocess(df)
        self.assertEqual(len(result), 0)
    
    @patch('preprocessing.load_spacy_model')
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    @patch('preprocessing.remove_words_dict')
    @patch('preprocessing.calculate_idf')
    def test_custom_text_column(self, mock_idf, mock_remove_words,
                                mock_remove_num, mock_punctuation,
                                mock_preprocess, mock_load):
        """Test with custom text column."""
        # Setup mocks
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        mock_remove_words.side_effect = lambda text, words: text
        mock_idf.return_value = [(0.5, 'the')]
        
        df = create_sample_dataframe('custom_column')
        result = batch_preprocess(df, text_column='custom_column', copy_dataframe=True)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Final_Processed', result.columns)


# ============================================================================
# Tests for pre_processing_routine
# ============================================================================

class TestPreProcessingRoutine(unittest.TestCase):
    
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    def test_basic_preprocessing(self, mock_remove_num, mock_punctuation, mock_preprocess):
        """Test basic preprocessing routine."""
        # Setup mocks
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        
        df = create_sample_dataframe()
        processed, processed_numberless = pre_processing_routine(df)
        
        self.assertIsInstance(processed, list)
        self.assertIsInstance(processed_numberless, list)
        self.assertEqual(len(processed), len(df))
        self.assertEqual(len(processed_numberless), len(df))
    
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    def test_returns_two_versions(self, mock_remove_num, mock_punctuation, mock_preprocess):
        """Test that function returns two versions."""
        # Setup mocks
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        
        df = create_sample_dataframe()
        result = pre_processing_routine(df)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
    
    def test_invalid_dataframe(self):
        """Test invalid DataFrame."""
        with self.assertRaises(ValueError):
            pre_processing_routine("not a dataframe")
    
    def test_missing_column(self):
        """Test missing column."""
        df = pd.DataFrame({'wrong_column': ['test']})
        with self.assertRaises(ValueError):
            pre_processing_routine(df)
    
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    def test_custom_text_column(self, mock_remove_num, mock_punctuation, mock_preprocess):
        """Test with custom text column."""
        # Setup mocks
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        
        df = create_sample_dataframe('custom_column')
        processed, processed_numberless = pre_processing_routine(df, text_column='custom_column')
        
        self.assertIsInstance(processed, list)
        self.assertIsInstance(processed_numberless, list)
        self.assertEqual(len(processed), len(df))


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(unittest.TestCase):
    
    @patch('preprocessing.load_spacy_model')
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    @patch('preprocessing.remove_words_dict')
    @patch('preprocessing.calculate_idf')
    def test_full_pipeline_integration(self, mock_idf, mock_remove_words,
                                       mock_remove_num, mock_punctuation,
                                       mock_preprocess, mock_load):
        """Test full preprocessing pipeline integration."""
        # Setup mocks
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        mock_remove_words.side_effect = lambda text, words: text
        mock_idf.return_value = [(0.5, 'the')]
        
        # Create test data
        df = pd.DataFrame({
            'Narrative_long': [
                "The pilot don't like flying!",
                "Aircraft experienced issues.",
                "Test@#$ special chars"
            ]
        })
        
        # Run full pipeline
        result = batch_preprocess(df, copy_dataframe=True)
        
        # Verify all expected columns exist
        self.assertIn('processed', result.columns)
        self.assertIn('processed_num_Narr', result.columns)
        self.assertIn('processed_stopword_Narr', result.columns)
        self.assertIn('Final_Processed', result.columns)
    
    @patch('preprocessing.load_spacy_model')
    @patch('preprocessing.preprocess_text')
    @patch('preprocessing.handle_punctuation')
    @patch('preprocessing.remove_num')
    @patch('preprocessing.remove_words_dict')
    @patch('preprocessing.calculate_idf')
    def test_full_pipeline_custom_column(self, mock_idf, mock_remove_words,
                                         mock_remove_num, mock_punctuation,
                                         mock_preprocess, mock_load):
        """Test full preprocessing pipeline with custom column."""
        # Setup mocks
        mock_nlp = create_mock_spacy_model()
        mock_load.return_value = mock_nlp
        mock_preprocess.side_effect = lambda x: x.lower() if x else ""
        mock_punctuation.side_effect = lambda x: x
        mock_remove_num.side_effect = lambda x: x
        mock_remove_words.side_effect = lambda text, words: text
        mock_idf.return_value = [(0.5, 'the')]
        
        # Create test data with custom column name
        df = pd.DataFrame({
            'incident_description': [
                "The pilot don't like flying!",
                "Aircraft experienced issues.",
                "Test@#$ special chars"
            ]
        })
        
        # Run full pipeline with custom column
        result = batch_preprocess(df, text_column='incident_description', copy_dataframe=True)
        
        # Verify all expected columns exist
        self.assertIn('processed', result.columns)
        self.assertIn('processed_num_Narr', result.columns)
        self.assertIn('processed_stopword_Narr', result.columns)
        self.assertIn('Final_Processed', result.columns)
        
        # Verify original column still exists
        self.assertIn('incident_description', result.columns)


if __name__ == '__main__':
    unittest.main()