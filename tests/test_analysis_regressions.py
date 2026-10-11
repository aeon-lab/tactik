"""Regressions for missing text, spaCy model injection, and TF*DF scoring."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc

# Import utilities first: it must not depend on preprocessing's import order.
import utilities
from preprocessing import pre_processing_routine


class TestMissingText(unittest.TestCase):
    def test_real_preprocessing_preserves_rows_with_missing_text(self):
        frame = pd.DataFrame({'Narrative_long': [
            'Engine FIRE!', None, np.nan, pd.NA, '', 'Weather delay'
        ]})
        original = frame.copy(deep=True)
        processed, numberless = pre_processing_routine(frame)
        self.assertEqual(processed, ['engine fire', '', '', '', '', 'weather delay'])
        self.assertEqual(numberless, processed)
        pd.testing.assert_frame_equal(frame, original)

    def test_nonmissing_invalid_input_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'string or a missing value'):
            utilities.preprocess_text(123)


class TestSpacyHelpers(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank('en')
        self.doc = Doc(self.nlp.vocab, words=['The', 'engines', 'failed', '.'],
                       lemmas=['the', 'engine', 'fail', '.'],
                       pos=['DET', 'NOUN', 'VERB', 'PUNCT'])
        self.model = Mock(vocab=self.nlp.vocab, Defaults=self.nlp.Defaults,
                          return_value=self.doc)

    def test_tokenizer_accepts_model_and_lemmatizes(self):
        self.assertEqual(utilities.spacy_tokenizer('The engines failed.',
                                                  nlp_model=self.model), 'engine fail')

    def test_custom_stopwords_and_minimum_length(self):
        self.assertEqual(utilities.spacy_tokenizer(
            'The engines failed.', nlp_model=self.model,
            custom_stopwords={'the', 'engines'}, min_token_length=4), 'fail')

    def test_pos_counts_with_injected_model(self):
        self.assertEqual(utilities.extract_most_common_pos_tags(
            ['The engines failed.', '', 'The engines failed.'],
            top_n=2, nlp_model=self.model), [('DET', 2), ('NOUN', 2)])
        self.assertEqual(self.model.call_count, 2)

    def test_legacy_global_model_is_still_supported(self):
        with patch.object(utilities, 'nlp', self.model, create=True):
            self.assertEqual(utilities.spacy_tokenizer('The engines failed.'), 'engine fail')
            self.assertEqual(utilities.extract_most_common_pos_tags(['text'], 1), [('DET', 1)])

    def test_missing_model_raises_actionable_error(self):
        for helper, value in [(utilities.spacy_tokenizer, 'text'),
                              (utilities.extract_most_common_pos_tags, ['text'])]:
            with self.subTest(helper=helper.__name__):
                with self.assertRaisesRegex(RuntimeError, 'model provided'):
                    helper(value)


class TestDocumentFrequencyScoring(unittest.TestCase):
    def test_keyword_scores_match_when_shared_intermediates_are_reused(self):
        corpus = ['common common rare', 'common other', 'third rare']
        tf_documents = utilities.calculate_tf(corpus)
        mean_tf = utilities.calculate_mean_tf(tf_documents)
        idf_scores = utilities.calculate_idf(corpus)

        self.assertEqual(
            utilities.calculate_tf_corpus(corpus, 5),
            utilities.calculate_tf_corpus(corpus, 5,
                                          tf_documents=tf_documents, mean_tf=mean_tf))
        self.assertEqual(
            utilities.calculate_tfidf_corpus(corpus, 5),
            utilities.calculate_tfidf_corpus(corpus, 5,
                                             tf_documents=tf_documents, idf_scores=idf_scores))
        self.assertEqual(
            utilities.calculate_tfdf_corpus(corpus, 5),
            utilities.calculate_tfdf_corpus(corpus, 5,
                                            tf_documents=tf_documents, mean_tf=mean_tf))

    def test_common_terms_are_rewarded_instead_of_inverse_weighted(self):
        corpus = ['common rare', 'common other', 'common third']
        result = utilities.calculate_tfdf_corpus(corpus, 4)
        self.assertEqual(next(iter(result)), 'common')
        self.assertAlmostEqual(result['common'], 1.5)
        self.assertAlmostEqual(result['rare'], 1 / 6)
        self.assertEqual(utilities.calculate_tfidf_corpus(corpus, 4)['common'], 0.0)

    def test_df_counts_documents_once_and_mean_includes_empty_documents(self):
        corpus = (text for text in ['Common common rare', 'common', ''])
        result = utilities.calculate_tfdf_corpus(corpus, 2)
        self.assertAlmostEqual(result['common'], 10 / 9)
        self.assertAlmostEqual(result['rare'], 1 / 9)

    def test_empty_documents_have_no_keywords(self):
        self.assertEqual(utilities.calculate_tfdf_corpus(['', ''], 5), {})


if __name__ == '__main__':
    unittest.main()
