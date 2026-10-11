"""Tests for lazy NLTK Punkt data handling."""

import importlib
import unittest
from unittest.mock import patch

import nltk_resources


class TestPunktResourceHandling(unittest.TestCase):
    def test_import_does_not_download_punkt(self):
        with patch.object(nltk_resources.nltk, 'download') as download:
            importlib.reload(nltk_resources)
        download.assert_not_called()

    @patch('nltk_resources.nltk.download')
    @patch('nltk_resources.nltk.data.find')
    def test_available_resource_does_not_download(self, find, download):
        self.assertTrue(nltk_resources.ensure_punkt_data())
        find.assert_called_once_with(nltk_resources._required_punkt_resources()[0][1])
        download.assert_not_called()

    @patch('nltk_resources.nltk.download', return_value=True)
    @patch('nltk_resources.nltk.data.find')
    def test_missing_resource_downloads_on_demand(self, find, download):
        find.side_effect = [LookupError('missing'), None]
        package, _ = nltk_resources._required_punkt_resources()[0]

        self.assertTrue(nltk_resources.ensure_punkt_data())

        download.assert_called_once_with(package, quiet=True)
        self.assertEqual(find.call_count, 2)

    @patch('nltk_resources.nltk.download')
    @patch('nltk_resources.nltk.data.find', side_effect=LookupError('missing'))
    def test_offline_mode_explains_how_to_install_resource(self, find, download):
        package, _ = nltk_resources._required_punkt_resources()[0]
        with self.assertRaisesRegex(LookupError, f"nltk.download\\('{package}'\\)"):
            nltk_resources.ensure_punkt_data(download=False)
        download.assert_not_called()

    @patch('nltk_resources.nltk.download', return_value=False)
    @patch('nltk_resources.nltk.data.find', side_effect=LookupError('missing'))
    def test_download_failure_is_actionable(self, find, download):
        package, _ = nltk_resources._required_punkt_resources()[0]
        with self.assertRaisesRegex(LookupError, f"Could not download.*{package}"):
            nltk_resources.ensure_punkt_data()

    @patch('nltk_resources._nltk_word_tokenize', return_value=['Hello', '.'])
    @patch('nltk_resources.ensure_punkt_data')
    def test_word_tokenize_checks_resources_on_demand(self, ensure, tokenize):
        result = nltk_resources.word_tokenize('Hello.')
        self.assertEqual(result, ['Hello', '.'])
        ensure.assert_called_once_with(download=True, language='english')
        tokenize.assert_called_once_with('Hello.', language='english', preserve_line=False)

    @patch('nltk_resources._nltk_word_tokenize', return_value=['Hello', '.'])
    @patch('nltk_resources.ensure_punkt_data')
    def test_preserve_line_does_not_check_punkt(self, ensure, tokenize):
        nltk_resources.word_tokenize('Hello.', preserve_line=True)
        ensure.assert_not_called()
        tokenize.assert_called_once_with('Hello.', language='english', preserve_line=True)


if __name__ == '__main__':
    unittest.main()
