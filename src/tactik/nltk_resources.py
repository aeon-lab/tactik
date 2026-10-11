"""On-demand helpers for NLTK tokenizer data."""

import nltk
from nltk.tokenize import word_tokenize as _nltk_word_tokenize


def _required_punkt_resources(language='english'):
    """Return the Punkt data package required by the installed NLTK version."""
    version = tuple(int(part) for part in nltk.__version__.split('.')[:2])
    # NLTK 3.9 switched to the pickle-free punkt_tab data format.
    if version >= (3, 9):
        return [('punkt_tab', f'tokenizers/punkt_tab/{language}')]
    return [('punkt', 'tokenizers/punkt')]


def ensure_punkt_data(download=True, language='english'):
    """Ensure Punkt data is available without downloading during module import.

    Args:
        download: Download missing NLTK data packages when True. Set False for
            offline environments to receive an actionable LookupError.
        language: Punkt language whose data should be checked.

    Returns:
        True when the required resources are available.
    """
    for package, resource_path in _required_punkt_resources(language):
        try:
            nltk.data.find(resource_path)
            continue
        except LookupError as error:
            if not download:
                raise LookupError(
                    f"NLTK resource '{package}' is missing. Install it with "
                    f"nltk.download('{package}') or call "
                    "ensure_punkt_data(download=True) when network access is available."
                ) from error

        try:
            downloaded = nltk.download(package, quiet=True)
        except Exception as error:
            raise LookupError(
                f"Could not download NLTK resource '{package}'. Install it with "
                f"nltk.download('{package}') when network access is available."
            ) from error
        if not downloaded:
            raise LookupError(
                f"Could not download NLTK resource '{package}'. Install it with "
                f"nltk.download('{package}') and check the environment's network access."
            )
        try:
            nltk.data.find(resource_path)
        except LookupError as error:
            raise LookupError(
                f"NLTK reported that '{package}' downloaded, but resource "
                f"'{resource_path}' is still unavailable."
            ) from error
    return True


def word_tokenize(text, language='english', preserve_line=False, *, download=True):
    """Tokenize text and obtain Punkt data on demand when needed.

    Set ``download=False`` to prevent network access and get an actionable
    error when the required NLTK resource has not been installed.
    """
    if not preserve_line:
        ensure_punkt_data(download=download, language=language)
    return _nltk_word_tokenize(text, language=language, preserve_line=preserve_line)
