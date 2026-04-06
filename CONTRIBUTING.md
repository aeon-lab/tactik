# Contributing to TACTIK

Thank you for your interest in contributing to TACTIK (Text Analysis, Clustering, Tuning, Information and Keyword Extraction). This document covers everything you need to know to contribute effectively — from setting up your environment to submitting a pull request.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Running the Tests](#running-the-tests)
- [Writing Tests](#writing-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Project Overview

TACTIK is a Python library for end-to-end text clustering and analysis. It is structured around six core pillars, each encapsulated in its own module:

| Pillar | Module | Responsibility |
|---|---|---|
| Preprocessing | `preprocessing.py` | Text cleaning, tokenisation, stopword removal |
| Embeddings | `embeddings.py` | TF-IDF, SBERT, Word2Vec, GloVe, LSA, UMAP, t-SNE |
| Clustering | `clustering_tuning.py` | UMAP + HDBSCAN pipeline, hyperparameter tuning |
| Topic Modelling & Labelling | `topic_extraction.py` | LDA topic modelling, BERT-based designator matching |
| Visualization | `visualization.py` | Cluster plotting and annotations |
| Orchestration | `clustering_pipeline.py` | High-level pipeline coordinating all other modules |

Contributions to any of these modules are welcome, as are improvements to tests, documentation, and CI configuration.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [Anaconda](https://www.anaconda.com/) or a standard Python virtual environment
- Git

### Setting Up Your Environment

1. Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/<your-username>/tactik.git
cd tactik
```

2. Create and activate a conda environment:

```bash
conda create -n tactik-dev python=3.11
conda activate tactik-dev
```

3. Install the package in editable mode along with all dependencies:

```bash
pip install -e .
```

4. Set the `PYTHONPATH` so the test runner can find the source modules directly:

```bash
conda env config vars set PYTHONPATH=/absolute/path/to/tactik/src/tactik
conda activate tactik-dev
```

Replace `/absolute/path/to/tactik` with the actual path to your cloned repository.

5. Verify the setup by running the test suite:

```bash
python -W ignore -m unittest discover -s tests
```

You should see all tests passing with a small number of intentional skips on Windows (t-SNE threadpool tests).

---

## Project Structure

```
tactik/
├── src/
│   └── tactik/
│       ├── __init__.py
│       ├── clustering_pipeline.py   # Orchestration layer
│       ├── clustering_tuning.py     # UMAP + HDBSCAN tuning
│       ├── embeddings.py            # Text embedding methods
│       ├── preprocessing.py         # Text preprocessing
│       ├── topic_extraction.py      # LDA + BERT topic labelling
│       ├── utilities.py             # Shared utility functions
│       └── visualization.py         # Cluster visualisation
├── tests/
│   ├── __init__.py                  # Adds src/tactik to sys.path
│   ├── test_clustering_pipeline.py
│   ├── test_clustering_tuning.py
│   ├── test_embeddings.py
│   ├── test_preprocessing.py
│   ├── test_topic_extraction.py
│   └── test_utilities.py
├── pyproject.toml
└── README.md
```

### Import Convention

All modules use a try/except import pattern to support both package-level imports (when installed) and direct imports (when running tests standalone):

```python
try:
    from .utilities import some_function
except ImportError:
    from utilities import some_function
```

Any new internal imports you add should follow this same pattern.

---

## Development Workflow

1. Create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. Make your changes, keeping commits focused and atomic.

3. Run the full test suite before pushing:

```bash
python -W ignore -m unittest discover -s tests
```

4. Push your branch and open a pull request against `main`.

---

## Running the Tests

TACTIK uses Python's built-in `unittest` framework. All tests live in the `tests/` directory.

### Run all tests

```bash
python -W ignore -m unittest discover -s tests
```

### Run a specific test file

```bash
python -W ignore -m unittest tests/test_preprocessing.py
```

### Run a specific test class or method

```bash
python -W ignore -m unittest tests.test_preprocessing.TestDefineStopwords
python -W ignore -m unittest tests.test_preprocessing.TestDefineStopwords.test_custom_text_column
```

### Known skips

Six tests in `test_embeddings.py` are skipped on Windows due to known threadpool conflicts with t-SNE. These tests run normally on Linux and macOS and are part of the CI suite.

### CI

Tests run automatically on GitHub Actions on every push and pull request. The CI environment uses the latest supported versions of all dependencies. If a test passes locally but fails in CI, the most common cause is a dependency version difference — in particular, scikit-learn 1.5+ renamed certain parameters (e.g. `n_iter` → `max_iter` in `TSNE`). Always test against the dependency versions specified in `pyproject.toml`.

---

## Writing Tests

Every new function or behaviour change must be accompanied by tests. TACTIK's test suite makes heavy use of `unittest.mock` to isolate units from their dependencies — follow the same pattern in the relevant test file.

### Key conventions

**Use `create_sample_dataframe()` for DataFrame fixtures.** The helper is defined at the top of each test file and creates a small DataFrame that includes a `None` row to exercise null-handling code paths. Do not assume all values in a text column are strings.

**Mock lambdas must handle non-string values.** When mocking `preprocess_text` or similar text functions, always use `isinstance` rather than a bare truthiness check:

```python
# Correct
mock_preprocess.side_effect = lambda x: x.lower() if isinstance(x, str) else ""

# Wrong — crashes on float NaN
mock_preprocess.side_effect = lambda x: x.lower() if x else ""
```

**Assert the full call signature when testing method calls.** If a function passes `random_state` or other parameters through to a dependency, include those in your `assert_called_once_with` check:

```python
mock_modeler.train_lda.assert_called_once_with(num_topics=10, passes=20, random_state=42)
```

**Patch at the right level.** Patch functions at the module where they are used, not where they are defined. For example, to mock `calculate_idf` inside `preprocessing.py`, use `@patch('preprocessing.calculate_idf')`.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for formatting.
- All public functions must have a NumPy-style or Google-style docstring covering parameters, return values, and exceptions.
- Type hints are expected for all function signatures.
- Do not use mutable default arguments. Use `None` as the default and initialise inside the function body.
- Keep functions focused. If a function is doing more than one thing, consider splitting it.
- Internal helper functions should be prefixed with a single underscore (e.g. `_apply_umap_hdbscan`).

---

## Submitting a Pull Request

Before opening a pull request, please ensure:

- All existing tests pass locally.
- New tests cover any new behaviour or bug fix.
- Docstrings are updated for any changed function signatures.
- The branch is up to date with `main`.

In your pull request description, briefly explain what the change does and why, and reference any related issues with `Fixes #<issue-number>` if applicable.

---

## Reporting Issues

If you encounter a bug or unexpected behaviour, please open a GitHub issue and include:

- A minimal reproducible example
- The Python and dependency versions you are using (`pip show tactik scikit-learn spacy`)
- The full traceback if an exception is raised

For questions about the codebase or design decisions, feel free to open a discussion on GitHub.
