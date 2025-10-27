import unittest
from unittest.mock import patch, MagicMock, call
import math
import pandas as pd
from collections import Counter
import numpy as np
import string

# Import the functions to test
from utilities import (
    extract_features,
    calculate_idf,
    calculate_tf,
    calculate_mean_tf,
    handle_punctuation,
    remove_num,
    preprocess_text,
    remove_words_dict,
    get_max_n_dict,
    calculate_tf_corpus,
    calculate_tfidf_corpus,
    calculate_tfdf_corpus,
    get_number_unique_words,
    calculate_word_length_distribution,
    calculate_lexical_diversity
)


class TestExtractFeatures(unittest.TestCase):
    """Comprehensive tests for extract_features function."""
    
    def test_basic_unigrams(self):
        """Test unigram extraction."""
        result = extract_features("hello world test", n=1)
        expected = {('hello',), ('world',), ('test',)}
        self.assertEqual(result, expected)
    
    def test_basic_bigrams(self):
        """Test bigram extraction."""
        result = extract_features("hello world test", n=2)
        expected = {
            ('hello',), ('world',), ('test',),
            ('hello', 'world'), ('world', 'test')
        }
        self.assertEqual(result, expected)
    
    def test_basic_trigrams(self):
        """Test trigram extraction."""
        result = extract_features("a b c d", n=3)
        expected = {
            ('a',), ('b',), ('c',), ('d',),
            ('a', 'b'), ('b', 'c'), ('c', 'd'),
            ('a', 'b', 'c'), ('b', 'c', 'd')
        }
        self.assertEqual(result, expected)
    
    def test_empty_string(self):
        """Test with empty string."""
        result = extract_features("", n=2)
        self.assertEqual(result, set())
    
    def test_single_word(self):
        """Test with single word."""
        result = extract_features("hello", n=3)
        expected = {('hello',)}
        self.assertEqual(result, expected)
    
    def test_two_words_with_large_n(self):
        """Test n-grams where n > number of words."""
        result = extract_features("hello world", n=5)
        expected = {('hello',), ('world',), ('hello', 'world')}
        self.assertEqual(result, expected)
    
    def test_case_sensitivity(self):
        """Test case normalization."""
        result = extract_features("Hello HELLO hello", n=1)
        expected = {('hello',)}
        self.assertEqual(result, expected)
    
    def test_duplicate_words(self):
        """Test with duplicate words."""
        result = extract_features("the cat the dog the", n=2)
        # Should contain unique n-grams only
        self.assertIn(('the',), result)
        self.assertIn(('cat',), result)
        self.assertIn(('the', 'cat'), result)
        self.assertIn(('cat', 'the'), result)
        self.assertIn(('the', 'dog'), result)
        self.assertIn(('dog', 'the'), result)
    
    def test_whitespace_handling(self):
        """Test with extra whitespace."""
        result = extract_features("hello   world", n=1)
        expected = {('hello',), ('world',)}
        self.assertEqual(result, expected)
    
    def test_unicode_characters(self):
        """Test with unicode characters."""
        result = extract_features("café résumé naïve", n=1)
        expected = {('café',), ('résumé',), ('naïve',)}
        self.assertEqual(result, expected)
    
    def test_n_equals_zero(self):
        """Test with n=0 (edge case)."""
        result = extract_features("hello world", n=0)
        self.assertEqual(result, set())
    
    def test_very_long_text(self):
        """Test with very long text."""
        long_text = " ".join([f"word{i}" for i in range(1000)])
        result = extract_features(long_text, n=2)
        # Should contain 1000 unigrams and 999 bigrams
        self.assertGreaterEqual(len(result), 1999)


class TestCalculateIDF(unittest.TestCase):
    """Comprehensive tests for calculate_idf function."""
    
    def test_basic_idf(self):
        """Test basic IDF calculation."""
        narratives = ["the cat sat", "the dog ran", "a bird flew"]
        result = calculate_idf(narratives)
        idf_dict = {word: score for score, word in result}
        
        # "the" appears in 2/3 documents
        self.assertAlmostEqual(idf_dict['the'], math.log(3/2), places=5)
        
        # Words in 1 document only
        self.assertAlmostEqual(idf_dict['cat'], math.log(3/1), places=5)
        self.assertAlmostEqual(idf_dict['bird'], math.log(3/1), places=5)
    
    def test_empty_narratives(self):
        """Test with empty list."""
        result = calculate_idf([])
        self.assertEqual(result, [])
    
    def test_single_narrative(self):
        """Test with single document."""
        result = calculate_idf(["hello world"])
        # All words should have IDF = log(1/1) = 0
        for score, word in result:
            self.assertEqual(score, 0.0)
    
    def test_all_unique_words(self):
        """Test when all words are unique across documents."""
        narratives = ["apple", "banana", "cherry"]
        result = calculate_idf(narratives)
        idf_dict = {word: score for score, word in result}
        
        # All words appear in 1/3 documents
        for word in ['apple', 'banana', 'cherry']:
            self.assertAlmostEqual(idf_dict[word], math.log(3/1), places=5)
    
    def test_all_same_words(self):
        """Test when all documents contain same words."""
        narratives = ["hello world", "hello world", "hello world"]
        result = calculate_idf(narratives)
        idf_dict = {word: score for score, word in result}
        
        # Words appear in 3/3 documents
        self.assertAlmostEqual(idf_dict['hello'], math.log(3/3), places=5)
        self.assertEqual(idf_dict['hello'], 0.0)
    
    def test_empty_strings_in_list(self):
        """Test with empty strings in narratives."""
        narratives = ["hello world", "", "test"]
        result = calculate_idf(narratives)
        idf_dict = {word: score for score, word in result}
        
        self.assertIn('hello', idf_dict)
        self.assertIn('test', idf_dict)
    
    def test_case_sensitivity(self):
        """Test case handling in IDF."""
        narratives = ["Hello world", "HELLO test", "hello again"]
        result = calculate_idf(narratives)
        idf_dict = {word: score for score, word in result}
        
        # "hello" should appear in all 3 documents (case-insensitive)
        self.assertEqual(idf_dict['hello'], 0.0)
    
    def test_result_format(self):
        """Test that result is properly formatted."""
        narratives = ["cat dog"]
        result = calculate_idf(narratives)
        
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], float)
            self.assertIsInstance(item[1], str)
    
    def test_sorted_result(self):
        """Test that result is sorted by IDF score."""
        narratives = ["a b c", "a b", "a"]
        result = calculate_idf(narratives)
        
        
        scores = [score for score, word in result]
        self.assertEqual(scores, sorted(scores, reverse=False))


class TestCalculateTF(unittest.TestCase):
    """Comprehensive tests for calculate_tf function."""
    
    def test_basic_tf(self):
        """Test basic TF calculation."""
        narratives = ["hello world hello", "world test"]
        result = calculate_tf(narratives)
        
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]['hello'], 2/3, places=5)
        self.assertAlmostEqual(result[0]['world'], 1/3, places=5)
        self.assertEqual(result[1]['world'], 0.5)
        self.assertEqual(result[1]['test'], 0.5)
    
    def test_empty_narratives(self):
        """Test with empty list."""
        result = calculate_tf([])
        self.assertEqual(result, [])
    
    def test_empty_string_in_list(self):
        """Test with empty string."""
        result = calculate_tf(["hello world", ""])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], {})
    
    def test_single_word_document(self):
        """Test document with single word."""
        result = calculate_tf(["hello"])
        self.assertEqual(result[0]['hello'], 1.0)
    
    def test_all_same_word(self):
        """Test document with repeated same word."""
        result = calculate_tf(["test test test test"])
        self.assertEqual(result[0]['test'], 1.0)
    
    def test_tf_sum_equals_one(self):
        """Test that TF values sum to 1 for each document."""
        narratives = ["a b c d e", "test word example"]
        result = calculate_tf(narratives)
        
        for tf_dict in result:
            total = sum(tf_dict.values())
            self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_case_insensitive(self):
        """Test case insensitive TF calculation."""
        result = calculate_tf(["Hello HELLO hello"])
        self.assertEqual(result[0]['hello'], 1.0)
    
    def test_unicode_handling(self):
        """Test with unicode characters."""
        result = calculate_tf(["café café résumé"])
        self.assertAlmostEqual(result[0]['café'], 2/3, places=5)
        self.assertAlmostEqual(result[0]['résumé'], 1/3, places=5)


class TestCalculateMeanTF(unittest.TestCase):
    """Comprehensive tests for calculate_mean_tf function."""
    
    def test_basic_mean_tf(self):
        """Test basic mean TF calculation."""
        tf_dicts = [
            {'word1': 0.5, 'word2': 0.5},
            {'word1': 0.3, 'word3': 0.7},
            {'word2': 1.0}
        ]
        result = calculate_mean_tf(tf_dicts)
        
        self.assertAlmostEqual(result['word1'], (0.5 + 0.3 + 0) / 3, places=5)
        self.assertAlmostEqual(result['word2'], (0.5 + 0 + 1.0) / 3, places=5)
        self.assertAlmostEqual(result['word3'], 0.7 / 3, places=5)
    
    def test_empty_list(self):
        """Test with empty list."""
        result = calculate_mean_tf([])
        self.assertEqual(result, {})
    
    def test_single_document(self):
        """Test with single document."""
        tf_dicts = [{'word1': 0.5, 'word2': 0.5}]
        result = calculate_mean_tf(tf_dicts)
        
        self.assertEqual(result['word1'], 0.5)
        self.assertEqual(result['word2'], 0.5)
    
    def test_all_empty_dicts(self):
        """Test with all empty dictionaries."""
        result = calculate_mean_tf([{}, {}, {}])
        self.assertEqual(result, {})
    
    def test_missing_values_as_zero(self):
        """Test that missing values are treated as 0."""
        tf_dicts = [
            {'word1': 1.0},
            {'word2': 1.0},
            {'word3': 1.0}
        ]
        result = calculate_mean_tf(tf_dicts)
        
        # Each word appears once, so mean = 1/3
        for word in ['word1', 'word2', 'word3']:
            self.assertAlmostEqual(result[word], 1/3, places=5)
    
    def test_consistent_words_across_docs(self):
        """Test when same words appear in all documents."""
        tf_dicts = [
            {'word1': 0.6, 'word2': 0.4},
            {'word1': 0.7, 'word2': 0.3},
            {'word1': 0.5, 'word2': 0.5}
        ]
        result = calculate_mean_tf(tf_dicts)
        
        self.assertAlmostEqual(result['word1'], (0.6 + 0.7 + 0.5) / 3, places=5)
        self.assertAlmostEqual(result['word2'], (0.4 + 0.3 + 0.5) / 3, places=5)


class TestHandlePunctuation(unittest.TestCase):
    """Comprehensive tests for handle_punctuation function."""
    
    def test_basic_punctuation_removal(self):
        """Test basic punctuation removal."""
        result = handle_punctuation("Hello, world! How are you?")
        expected = "Hello  world  How are you "
        self.assertEqual(result, expected)
    
    def test_no_punctuation(self):
        """Test text without punctuation."""
        result = handle_punctuation("hello world")
        self.assertEqual(result, "hello world")
    
    def test_empty_string(self):
        """Test empty string."""
        result = handle_punctuation("")
        self.assertEqual(result, "")
    
    def test_only_punctuation(self):
        """Test string with only punctuation."""
        result = handle_punctuation("!@#$%^&*()")
        # All punctuation should be replaced with spaces
        self.assertTrue(all(c == ' ' for c in result))
    
    def test_all_punctuation_types(self):
        """Test various punctuation marks."""
        punctuation = string.punctuation
        text = f"word{punctuation}test"
        result = handle_punctuation(text)
        
        # Check that no punctuation remains
        for p in punctuation:
            self.assertNotIn(p, result)
    
    def test_multiple_consecutive_punctuation(self):
        """Test multiple consecutive punctuation marks."""
        result = handle_punctuation("Hello!!! World???")
        self.assertNotIn('!', result)
        self.assertNotIn('?', result)
    
    def test_unicode_punctuation(self):
        """Test unicode punctuation (may depend on implementation)."""
        result = handle_punctuation("Hello… world—test")
        # Basic test - implementation may vary
        self.assertIn('Hello', result)
        self.assertIn('world', result)


class TestRemoveNum(unittest.TestCase):
    """Comprehensive tests for remove_num function."""
    
    def test_basic_digit_removal(self):
        """Test basic digit removal."""
        result = remove_num("Hello123world456")
        expected = "Hello   world   "
        self.assertEqual(result, expected)
    
    def test_no_digits(self):
        """Test text without digits."""
        result = remove_num("hello world")
        self.assertEqual(result, "hello world")
    
    def test_empty_string(self):
        """Test empty string."""
        result = remove_num("")
        self.assertEqual(result, "")
    
    def test_only_digits(self):
        """Test string with only digits."""
        result = remove_num("1234567890")
        self.assertTrue(all(c == ' ' for c in result))
    
    def test_mixed_digits_and_letters(self):
        """Test mixed content."""
        result = remove_num("abc123def456ghi")
        self.assertNotIn('1', result)
        self.assertNotIn('2', result)
        self.assertIn('abc', result)
        self.assertIn('def', result)
    
    def test_digits_at_boundaries(self):
        """Test digits at start and end."""
        result = remove_num("123hello456")
        self.assertIn('hello', result)
        self.assertNotIn('1', result)
        self.assertNotIn('4', result)


class TestPreprocessText(unittest.TestCase):
    """Comprehensive tests for preprocess_text function."""
    
    def test_comprehensive_preprocessing(self):
        """Test complete preprocessing pipeline."""
        text = "Hello, World! This is a TEST 123."
        result = preprocess_text(text)
        expected = "hello world this is a test"
        self.assertEqual(result, expected)
    
    def test_multiple_spaces_normalization(self):
        """Test multiple spaces are normalized."""
        text = "hello    world   test"
        result = preprocess_text(text)
        expected = "hello world test"
        self.assertEqual(result, expected)
    
    def test_empty_string(self):
        """Test empty string."""
        result = preprocess_text("")
        self.assertEqual(result, "")
    
    def test_only_special_characters(self):
        """Test string with only special characters."""
        result = preprocess_text("!@#123$%^")
        self.assertEqual(result, "")
    
    def test_mixed_case(self):
        """Test case normalization."""
        result = preprocess_text("HeLLo WoRLd")
        self.assertEqual(result, "hello world")
    
    def test_leading_trailing_spaces(self):
        """Test leading and trailing whitespace removal."""
        result = preprocess_text("   hello world   ")
        self.assertEqual(result, "hello world")
    
    def test_tabs_and_newlines(self):
        """Test tab and newline normalization."""
        result = preprocess_text("hello\tworld\ntest")
        # Should be normalized to single spaces
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("test", result)
    
    def test_unicode_preservation(self):
        """Test that unicode characters are preserved."""
        result = preprocess_text("café résumé")
        self.assertIn("café", result)
        self.assertIn("résumé", result)


class TestRemoveWordsDict(unittest.TestCase):
    """Comprehensive tests for remove_words_dict function."""
    
    def test_basic_word_removal(self):
        """Test basic word removal."""
        narrative = "the quick brown fox jumps"
        dictionary = ["the", "fox"]
        result = remove_words_dict(narrative, dictionary)
        expected = "quick brown jumps"
        self.assertEqual(result, expected)
    
    def test_empty_dictionary(self):
        """Test with empty dictionary."""
        result = remove_words_dict("hello world", [])
        self.assertEqual(result, "hello world")
    
    def test_remove_all_words(self):
        """Test removing all words."""
        result = remove_words_dict("hello world", ["hello", "world"])
        self.assertEqual(result, "")
    
    def test_empty_narrative(self):
        """Test with empty narrative."""
        result = remove_words_dict("", ["word"])
        self.assertEqual(result, "")
        
    def test_partial_word_match(self):
        """Test that partial words are not removed."""
        result = remove_words_dict("hello helloworld", ["hello"])
        # "helloworld" should not be removed
        self.assertIn("helloworld", result)
        self.assertNotIn("hello ", result)
    
    def test_multiple_occurrences(self):
        """Test removing word that appears multiple times."""
        result = remove_words_dict("the cat the dog the", ["the"])
        self.assertEqual(result, "cat dog")
    
    def test_word_not_in_narrative(self):
        """Test removing word that doesn't exist."""
        result = remove_words_dict("hello world", ["absent"])
        self.assertEqual(result, "hello world")


class TestGetMaxNDict(unittest.TestCase):
    """Comprehensive tests for get_max_n_dict function."""
    
    def test_basic_top_n(self):
        """Test basic top N extraction."""
        test_dict = {'a': 10, 'b': 5, 'c': 15, 'd': 3}
        result = get_max_n_dict(test_dict, 2)
        expected_keys = ['c', 'a']
        self.assertEqual(list(result.keys()), expected_keys)
        self.assertEqual(result['c'], 15)
        self.assertEqual(result['a'], 10)
    
    def test_n_larger_than_dict(self):
        """Test n larger than dictionary size."""
        result = get_max_n_dict({'a': 1, 'b': 2}, 5)
        self.assertEqual(len(result), 2)
    
    def test_n_equals_zero(self):
        """Test n = 0."""
        result = get_max_n_dict({'a': 1, 'b': 2}, 0)
        self.assertEqual(result, {})
    
    def test_empty_dict(self):
        """Test empty dictionary."""
        result = get_max_n_dict({}, 5)
        self.assertEqual(result, {})
    
    def test_n_equals_dict_size(self):
        """Test n equals dictionary size."""
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        result = get_max_n_dict(test_dict, 3)
        self.assertEqual(len(result), 3)
    
    def test_tied_values(self):
        """Test with tied values."""
        test_dict = {'a': 5, 'b': 5, 'c': 3}
        result = get_max_n_dict(test_dict, 2)
        # Should return 2 items, both with value 5
        self.assertEqual(len(result), 2)
        for val in result.values():
            self.assertGreaterEqual(val, 3)
    
    def test_negative_values(self):
        """Test with negative values."""
        test_dict = {'a': -5, 'b': -10, 'c': -3}
        result = get_max_n_dict(test_dict, 1)
        # Should return the largest (least negative)
        self.assertEqual(list(result.keys())[0], 'c')
    
    def test_type_error(self):
        """Test type error handling."""
        with self.assertRaises(TypeError):
            get_max_n_dict("not a dict", 5)
        
        with self.assertRaises(TypeError):
            get_max_n_dict(None, 5)
    
    def test_value_error_negative_n(self):
        """Test value error for negative n."""
        with self.assertRaises(ValueError):
            get_max_n_dict({'a': 1}, -1)
    
    def test_order_preservation(self):
        """Test that order is by value descending."""
        test_dict = {'a': 1, 'b': 5, 'c': 3, 'd': 10, 'e': 7}
        result = get_max_n_dict(test_dict, 3)
        keys = list(result.keys())
        values = list(result.values())
        
        # Check values are in descending order
        self.assertEqual(values, sorted(values, reverse=True))


class TestCalculateTFCorpus(unittest.TestCase):
    """Comprehensive tests for calculate_tf_corpus function."""
    
    def test_basic_tf_corpus(self):
        """Test basic TF corpus calculation."""
        corpus = ["hello world hello", "world test world"]
        result = calculate_tf_corpus(corpus, 2)
        
        self.assertIsInstance(result, dict)
        self.assertLessEqual(len(result), 2)
        self.assertIn('world', result)
    
    def test_empty_corpus(self):
        """Test with empty corpus."""
        result = calculate_tf_corpus([], 5)
        self.assertEqual(result, {})
    
    def test_single_document(self):
        """Test with single document."""
        corpus = ["hello world test"]
        result = calculate_tf_corpus(corpus, 2)
        self.assertEqual(len(result), 2)
    
    def test_n_larger_than_vocabulary(self):
        """Test n larger than vocabulary size."""
        corpus = ["hello world"]
        result = calculate_tf_corpus(corpus, 10)
        self.assertEqual(len(result), 2)  # Only 2 unique words
    
    def test_all_words_equal_frequency(self):
        """Test when all words have equal frequency."""
        corpus = ["a b c", "d e f"]
        result = calculate_tf_corpus(corpus, 3)
        # All should have same mean TF
        values = list(result.values())
        self.assertTrue(all(abs(v - values[0]) < 1e-5 for v in values))


class TestGetNumberUniqueWords(unittest.TestCase):
    """Comprehensive tests for get_number_unique_words function."""
    
    @patch('utilities.CountVectorizer')
    def test_basic_unique_words(self, mock_vectorizer):
        """Test basic unique word count."""
        mock_instance = MagicMock()
        mock_instance.get_feature_names_out.return_value = ['word1', 'word2', 'word3']
        mock_vectorizer.return_value = mock_instance
        
        narratives = ["hello world", "world test"]
        result = get_number_unique_words(narratives)
        
        self.assertEqual(result, 3)
        mock_instance.fit.assert_called_once_with(narratives)
    
    def test_none_input(self):
        """Test with None input."""
        with self.assertRaises(ValueError):
            get_number_unique_words(None)
    
    def test_empty_list(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            get_number_unique_words([])
    
    def test_non_string_element(self):
        """Test with non-string elements."""
        with self.assertRaises(ValueError):
            get_number_unique_words([123, "text"])
        
        with self.assertRaises(ValueError):
            get_number_unique_words(["text", None])
    
    @patch('utilities.CountVectorizer')
    def test_single_narrative(self, mock_vectorizer):
        """Test with single narrative."""
        mock_instance = MagicMock()
        mock_instance.get_feature_names_out.return_value = ['hello', 'world']
        mock_vectorizer.return_value = mock_instance
        
        result = get_number_unique_words(["hello world"])
        self.assertEqual(result, 2)


class TestCalculateWordLengthDistribution(unittest.TestCase):
    """Comprehensive tests for calculate_word_length_distribution function."""
    
    def test_basic_distribution(self):
        """Test basic word length distribution."""
        narratives = ["cat dog", "hello world test"]
        result = calculate_word_length_distribution(narratives)
        
        expected = {3: 2, 4: 1, 5: 2}
        self.assertEqual(result, expected)
    
    def test_empty_narratives(self):
        """Test with empty list."""
        result = calculate_word_length_distribution([])
        self.assertEqual(result, {})
    
    def test_empty_string_in_list(self):
        """Test with empty string."""
        result = calculate_word_length_distribution(["hello", ""])
        expected = {5: 1}
        self.assertEqual(result, expected)
    
    def test_single_word(self):
        """Test with single word."""
        result = calculate_word_length_distribution(["hello"])
        expected = {5: 1}
        self.assertEqual(result, expected)
    
    def test_all_same_length(self):
        """Test when all words have same length."""
        result = calculate_word_length_distribution(["cat dog fox"])
        expected = {3: 3}
        self.assertEqual(result, expected)
    
    def test_very_long_words(self):
        """Test with very long words."""
        long_word = "a" * 100
        result = calculate_word_length_distribution([long_word])
        expected = {100: 1}
        self.assertEqual(result, expected)
    
    def test_single_character_words(self):
        """Test with single character words."""
        result = calculate_word_length_distribution(["a b c"])
        expected = {1: 3}
        self.assertEqual(result, expected)
    
    def test_mixed_lengths(self):
        """Test with various word lengths."""
        narratives = ["a bb ccc dddd"]
        result = calculate_word_length_distribution(narratives)
        expected = {1: 1, 2: 1, 3: 1, 4: 1}
        self.assertEqual(result, expected)


class TestCalculateLexicalDiversity(unittest.TestCase):
    """Comprehensive tests for calculate_lexical_diversity function."""
    
    def test_basic_lexical_diversity(self):
        """Test basic lexical diversity calculation."""
        narratives = [
            "the cat sat on the mat",  # 5 unique / 6 total
            "hello hello world",       # 2 unique / 3 total
            "unique words here"        # 3 unique / 3 total
        ]
        result = calculate_lexical_diversity(narratives)
        
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 5/6, places=5)
        self.assertAlmostEqual(result[1], 2/3, places=5)
        self.assertEqual(result[2], 1.0)
    
    def test_empty_string_diversity(self):
        """Test lexical diversity with empty string."""
        result = calculate_lexical_diversity(["hello world", ""])
        self.assertEqual(result[1], 0.0)
    
    def test_single_word_diversity(self):
        """Test with single word (diversity = 1.0)."""
        result = calculate_lexical_diversity(["hello"])
        self.assertEqual(result[0], 1.0)
    
    def test_all_repeated_words(self):
        """Test when all words are the same."""
        result = calculate_lexical_diversity(["test test test test"])
        self.assertAlmostEqual(result[0], 0.25, places=5)  # 1 unique / 4 total
    
    def test_case_sensitive_option(self):
        """Test case sensitive vs insensitive."""
        # Case insensitive (default)
        result_insensitive = calculate_lexical_diversity(
            ["The the THE"], case_sensitive=False
        )
        self.assertAlmostEqual(result_insensitive[0], 1/3, places=5)
        
        # Case sensitive
        result_sensitive = calculate_lexical_diversity(
            ["The the THE"], case_sensitive=True
        )
        self.assertEqual(result_sensitive[0], 1.0)  # 3 unique / 3 total
    
    def test_punctuation_removal_option(self):
        """Test punctuation removal option."""
        result_with_punct = calculate_lexical_diversity(
            ["hello! hello."], remove_punctuation=True
        )
        self.assertEqual(result_with_punct[0], 0.5)
        
        result_without_punct = calculate_lexical_diversity(
            ["hello! hello."], remove_punctuation=False
        )
        # Should treat "hello!" and "hello." as different
        self.assertGreater(result_without_punct[0], result_with_punct[0])
    
    def test_multiple_narratives(self):
        """Test with multiple narratives."""
        narratives = ["a b c", "a a b", "x y z"]
        result = calculate_lexical_diversity(narratives)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 1.0)  # 3/3
        self.assertAlmostEqual(result[1], 2/3, places=5)  # 2/3
        self.assertEqual(result[2], 1.0)  # 3/3
    
    def test_none_input(self):
        """Test with None input."""
        with self.assertRaises(ValueError):
            calculate_lexical_diversity(None)
    
    def test_empty_list(self):
        """Test with empty list."""
        with self.assertRaises(ValueError):
            calculate_lexical_diversity([])
    
    def test_non_string_element(self):
        """Test with non-string elements."""
        with self.assertRaises(ValueError):
            calculate_lexical_diversity([123, "text"])
        
        with self.assertRaises(ValueError):
            calculate_lexical_diversity(["text", None, "more"])
    
    def test_very_long_text(self):
        """Test with very long text."""
        long_text = " ".join([f"word{i}" for i in range(1000)])
        result = calculate_lexical_diversity([long_text])
        self.assertEqual(result[0], 1.0)  # All unique words
    
    def test_whitespace_handling(self):
        """Test proper whitespace handling."""
        result = calculate_lexical_diversity(["hello    world   hello"])
        self.assertAlmostEqual(result[0], 2/3, places=5)


class TestCalculateTFIDFCorpus(unittest.TestCase):
    """Comprehensive tests for calculate_tfidf_corpus function."""
    
    def test_basic_tfidf(self):
        """Test basic TF-IDF calculation."""
        corpus = ["hello world", "world test", "hello test"]
        result = calculate_tfidf_corpus(corpus, 2)
        
        self.assertIsInstance(result, dict)
        self.assertLessEqual(len(result), 2)
        
        # All scores should be positive
        for score in result.values():
            self.assertGreater(score, 0)
    
    def test_empty_corpus(self):
        """Test with empty corpus."""
        result = calculate_tfidf_corpus([], 5)
        self.assertEqual(result, {})
    
    def test_single_document(self):
        """Test with single document."""
        corpus = ["hello world test"]
        result = calculate_tfidf_corpus(corpus, 2)
        self.assertEqual(len(result), 2)
    
    def test_n_larger_than_vocabulary(self):
        """Test n larger than vocabulary."""
        corpus = ["hello world"]
        result = calculate_tfidf_corpus(corpus, 10)
        self.assertLessEqual(len(result), 2)
    
    def test_common_vs_rare_words(self):
        """Test that rare words get higher TF-IDF scores."""
        corpus = [
            "common common common rare",
            "common test",
            "common example"
        ]
        result = calculate_tfidf_corpus(corpus, 3)
        
        # "rare" should have higher TF-IDF than "common"
        if 'rare' in result and 'common' in result:
            self.assertGreater(result['rare'], result['common'])
    
    def test_result_format(self):
        """Test result format and types."""
        corpus = ["hello world"]
        result = calculate_tfidf_corpus(corpus, 1)
        
        for key, value in result.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, (int, float))


class TestCalculateTFDFCorpus(unittest.TestCase):
    """Comprehensive tests for calculate_tfdf_corpus function."""
    
    def test_basic_tfdf(self):
        """Test basic TF-DF calculation."""
        corpus = ["hello world hello", "world test"]
        result = calculate_tfdf_corpus(corpus, 2)
        
        self.assertIsInstance(result, dict)
        self.assertLessEqual(len(result), 2)
    
    def test_empty_corpus(self):
        """Test with empty corpus."""
        result = calculate_tfdf_corpus([], 5)
        self.assertEqual(result, {})
    
    def test_single_document(self):
        """Test with single document."""
        corpus = ["hello world test"]
        result = calculate_tfdf_corpus(corpus, 2)
        self.assertEqual(len(result), 2)
    
    def test_n_zero(self):
        """Test with n=0."""
        corpus = ["hello world"]
        result = calculate_tfdf_corpus(corpus, 0)
        self.assertEqual(result, {})
    
    def test_repeated_words(self):
        """Test corpus with repeated words."""
        corpus = ["test test test", "test example"]
        result = calculate_tfdf_corpus(corpus, 2)
        
        # "test" appears frequently
        self.assertIn('test', result)
    
    def test_result_values_positive(self):
        """Test that all result values are positive or zero."""
        corpus = ["hello world", "test example"]
        result = calculate_tfdf_corpus(corpus, 3)
        
        for value in result.values():
            self.assertGreaterEqual(value, 0)


class TestEdgeCasesAndIntegration(unittest.TestCase):
    """Test edge cases and integration scenarios."""
    
    def test_unicode_throughout_pipeline(self):
        """Test unicode handling across multiple functions."""
        text = "café naïve résumé"
        
        # Test preprocessing
        processed = preprocess_text(text)
        self.assertIn("café", processed)
        
        # Test feature extraction
        features = extract_features(text, n=1)
        self.assertIn(('café',), features)
        
        # Test TF calculation
        tf_result = calculate_tf([text])
        self.assertIn('café', tf_result[0])
    
    def test_very_large_corpus(self):
        """Test with large corpus."""
        large_corpus = [f"document {i} with word{i}" for i in range(100)]
        
        result = calculate_idf(large_corpus)
        self.assertGreater(len(result), 0)
        
        tf_result = calculate_tf(large_corpus)
        self.assertEqual(len(tf_result), 100)
    
    def test_special_characters_handling(self):
        """Test handling of special characters throughout."""
        text = "test@example.com #hashtag $money 50%"
        processed = preprocess_text(text)
        
        # Should handle special characters appropriately
        self.assertIsInstance(processed, str)
    
    def test_mixed_empty_and_valid_documents(self):
        """Test corpus with mix of empty and valid documents."""
        corpus = ["hello world", "", "test", "", "example"]
        
        tf_result = calculate_tf(corpus)
        self.assertEqual(len(tf_result), 5)
        self.assertEqual(tf_result[1], {})
        self.assertEqual(tf_result[3], {})
    
    def test_consistency_across_functions(self):
        """Test consistency of preprocessing across different functions."""
        text = "Hello World"
        
        processed = preprocess_text(text)
        features = extract_features(text, n=1)
        
        # Both should lowercase
        processed_words = set(processed.split())
        feature_words = {f[0] for f in features}
        
        self.assertEqual(processed_words, feature_words)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small TF values
        corpus = ["a " + " ".join([f"word{i}" for i in range(1000)])]
        tf_result = calculate_tf(corpus)
        
        # Should handle small fractions
        self.assertAlmostEqual(sum(tf_result[0].values()), 1.0, places=5)
    
    def test_single_character_handling(self):
        """Test handling of single characters."""
        result = extract_features("a b c", n=1)
        self.assertEqual(len(result), 3)
        
        diversity = calculate_lexical_diversity(["a b c"])
        self.assertEqual(diversity[0], 1.0)
    
    def test_maximum_ngram_size(self):
        """Test with very large n-gram size."""
        text = "word1 word2 word3"
        result = extract_features(text, n=10)
        
        # Should only return n-grams up to the text length
        max_ngram_size = max(len(ngram) for ngram in result)
        self.assertEqual(max_ngram_size, 3)
    
    def test_duplicate_removal_in_features(self):
        """Test that features contain unique items only."""
        text = "test test test"
        result = extract_features(text, n=2)
        
        # Should have unique tuples
        self.assertEqual(len(result), len(set(result)))


class TestErrorHandlingAndValidation(unittest.TestCase):
    """Test error handling and input validation."""
    
    def test_invalid_types_extract_features(self):
        """Test extract_features with invalid types."""
        with self.assertRaises((TypeError, AttributeError)):
            extract_features(None, n=1)
        
        with self.assertRaises((TypeError, AttributeError)):
            extract_features(123, n=1)
    
    def test_invalid_n_parameter(self):
        """Test functions with invalid n parameter."""
        # Negative n
        result = extract_features("hello world", n=-1)
        # Should handle gracefully (empty set or error)
        self.assertIsInstance(result, set)
    
    def test_mixed_types_in_list(self):
        """Test lists with mixed types."""
        with self.assertRaises((TypeError, AttributeError)):
            calculate_tf(["hello", 123, None])
    
    def test_boundary_values(self):
        """Test boundary values for numeric parameters."""
        # n = 1 (minimum meaningful value)
        result = extract_features("hello", n=1)
        self.assertEqual(result, {('hello',)})
        
        # Empty dictionary with n=0
        result = get_max_n_dict({'a': 1}, 0)
        self.assertEqual(result, {})
    
    def test_malformed_tf_dicts(self):
        """Test calculate_mean_tf with malformed input."""
        # Non-numeric values should be handled or raise error
        try:
            result = calculate_mean_tf([{'word': 'not_a_number'}])
            # If it doesn't raise, check the result
            self.assertIsInstance(result, dict)
        except (TypeError, ValueError):
            # Expected to raise
            pass
    
    def test_whitespace_only_strings(self):
        """Test with whitespace-only strings."""
        result = extract_features("   \t\n  ", n=1)
        self.assertEqual(result, set())
        
        processed = preprocess_text("   \t\n  ")
        self.assertEqual(processed, "")
    
    def test_very_long_single_word(self):
        """Test with extremely long single word."""
        long_word = "a" * 10000
        result = extract_features(long_word, n=1)
        self.assertEqual(len(result), 1)
        
        diversity = calculate_lexical_diversity([long_word])
        self.assertEqual(diversity[0], 1.0)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance-related edge cases."""
    
    def test_large_vocabulary(self):
        """Test with very large vocabulary."""
        corpus = [" ".join([f"word{i}" for i in range(100)]) for _ in range(10)]
        
        result = calculate_tf_corpus(corpus, 50)
        self.assertLessEqual(len(result), 50)
    
    def test_many_documents(self):
        """Test with many documents."""
        corpus = [f"document {i}" for i in range(1000)]
        
        idf_result = calculate_idf(corpus)
        self.assertGreater(len(idf_result), 0)
    
    def test_repeated_calculations(self):
        """Test repeated calculations give consistent results."""
        corpus = ["hello world", "test example"]
        
        result1 = calculate_tf(corpus)
        result2 = calculate_tf(corpus)
        
        self.assertEqual(result1, result2)


if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)
    
    