<img src="https://raw.githubusercontent.com/npsAub/tactik/main/tactik_header.png" alt="TACTIK Header" width="800"/>

# TACTIK
## Text Analysis, Clustering, Tuning, Information and Keyword Extraction

Tactik started as a side project to streamline clustering of aviation-related reports. The pipeline initially faced long processing times and became a bottleneck for analysis. These issues were addressed, and further functionality was added to enable intuitive topic extraction. The pipeline was adapted to work domain-agnostically while keeping the core use case in mind. With this functionality, we decided to release the package publicly so other researchers can contribute to it, build on it, or benefit from the included tools. Thank you for using TACTIK — we hope you find it as useful as we did in our research!

## Features

- **End-to-End Clustering Pipeline**: Automated workflow from preprocessing to cluster analysis  
- **Modular Design**: Use the different components as standalone modules or full pipelines  
- **Layered Effective Methods**: UMAP dimensionality reduction + HDBSCAN clustering  
- **Hyperparameter Tuning**: Automated parameter optimization using random search  
- **Keyword Extraction**: Multiple methods including TF, TF-IDF, DF, and YAKE  
- **Topic Modeling**: LDA-based topic discovery with BERT-powered semantic matching *(still in development)*  
- **Rich Visualizations**: t-SNE plots with customizable styling and annotations  
- **Memory Efficient**: Optimized for large datasets with lazy evaluation and caching  

## Installation

```bash
# Install tactik
pip install tactik

# Or install from source
git clone https://github.com/npsAub/tactik.git
cd tactik
pip install -e .
```

### Dependencies

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
pip install umap-learn hdbscan gensim nltk yake
pip install transformers torch
```

## Core Components

### 1. ClusteringPipeline
Main orchestrator class that coordinates the entire analysis workflow.

**Key Methods:**
- `preprocess_data()` - Text cleaning and stopword removal  
- `cluster_data()` - Clustering with fixed parameters  
- `tune_and_cluster()` - Clustering with hyperparameter tuning  
- `cluster_and_extract_keywords()` - Integrated clustering + keyword extraction  
- `cluster_and_analyze_topics()` - Integrated clustering + topic modeling  
- `visualize_clusters()` - Create cluster visualizations  
- `get_cluster_summary()` - Generate cluster statistics  

### 2. Clustering & Tuning
Low-level clustering functions with hyperparameter optimization.

**Key Functions:**
- `tune_clustering_hyperparameters()` - Random search optimization  
- `apply_best_clustering()` - Apply optimized parameters  
- `full_clustering_pipeline()` - Complete pipeline with tuning  
- `full_clustering_pipeline_fixed_params()` - Pipeline with fixed parameters  

**Supported Metrics:**
- `davies_bouldin`: Lower is better (measures cluster separation)  
- `calinski_harabasz`: Higher is better (ratio of between/within cluster dispersion)  

### 3. Keyword Extraction
Extract representative keywords from each cluster using multiple methods.

**KeywordExtractor Class:**
- `extract_keywords_per_cluster()` - Extract keywords using multiple methods  
- `save_keywords()` - Save results to CSV  

**Extraction Methods:**
- **TF**: Term Frequency  
- **TF-IDF**: Term Frequency–Inverse Document Frequency  
- **TF-DF**: Term Frequency–Document Frequency  
- **YAKE**: Yet Another Keyword Extractor (long and short narratives)  

### 4. Topic Modeling
Discover latent topics using LDA. Topic labels are optional exploratory suggestions: supply domain-specific designators or explicitly select the built-in `aviation` profile. Matches below the configurable cosine-similarity threshold are left unmatched; the default threshold is 0.5 and should be calibrated for the domain and embedding model.

**TopicModeler Class:**
- `train_lda()` - Train Latent Dirichlet Allocation model  
- `get_cluster_topics()` - Get top topics per cluster  
- `match_designators_to_topics()` - Suggest designator matches using BERT embeddings; topics may remain unmatched
- `get_bert_embedding()` - Compute BERT embeddings for semantic matching  

**Built-in aviation profile designators** (available by passing `designator_profile="aviation"`):
- Inadequate or inaccurate knowledge  
- Poor judgment and decision-making  
- Failure to follow procedures  
- Poor communication  
- Inadequate monitoring or vigilance  
- Task management and prioritization  
- Stress and psychological factors  
- Physical or physiological factors  
- Technical or system failures  
- Environmental factors  

### 5. Visualization
Create publication-ready visualizations of clustering results.

**Visualization Functions:**
- `plot_clusters()` - Basic cluster scatter plot  
- `plot_clusters_with_annotations()` - Plot with category annotations  
- `plot_cluster_comparison()` - Side-by-side comparison plots  
- `set_visualization_style()` - Configure plot styling  
- `get_cluster_palette()` - Generate color palettes  
- `get_cluster_markers()` - Generate marker styles  

## Pipeline Architecture

```
Input Data (DataFrame)
    ↓
Preprocessing
    ├── Text cleaning
    ├── Stopword removal
    └── Tokenization
    ↓
Vectorization (TF-IDF)
    ↓
Dimensionality Reduction (UMAP)
    ↓
Clustering (HDBSCAN)
    ↓
Visualization (t-SNE)
    ↓
Analysis
    ├── Keyword Extraction
    └── Topic Modeling (LDA + BERT)
```

## Performance Considerations

### Memory Optimization
- DataFrame lazy copying  
- Vectorization caching  
- Sparse TF-IDF stays sparse through preprocessing and into UMAP
- Euclidean HDBSCAN on reduced coordinates avoids an explicit all-pairs distance matrix
- BERT embedding cache with `clear_cache()` method  
- Incremental topic probability calculations  

### GPU Acceleration
GPU acceleration is available for BERT computations when initializing TopicModeler with `use_gpu=True`.

### Large Datasets
For large datasets, consider:
- Disabling t-SNE computation with `compute_tsne=False`  
- Using fixed parameters instead of hyperparameter tuning  
- Clearing BERT embedding cache periodically  

### Sparse clustering and distance settings

The `ClusteringPipeline` methods use cosine distance for UMAP and Euclidean
distance for HDBSCAN by default. TF-IDF remains a sparse CSR matrix until UMAP
produces the small, dense embedding. Neither clustering pipeline converts the
document-vocabulary matrix to a dense array.

```python
from tactik.clustering_pipeline import ClusteringPipeline

pipeline = ClusteringPipeline(df, text_column="text")
pipeline.preprocess_data()
result = pipeline.cluster_data(
    umap_metric="cosine",
    hdbscan_metric="euclidean",
    standardize=False,
    compute_tsne=False,
)

# The tuned path accepts the same distance, scaling, and visualization settings.
tuned = pipeline.tune_and_cluster(
    umap_metric="cosine",
    hdbscan_metric="euclidean",
    max_iterations=20,
    compute_tsne=False,
)
```

`standardize=False` and `compute_tsne=False` are now the defaults for
`cluster_data()`, `tune_and_cluster()`, and the combined keyword/topic methods.
When scaling is enabled, sparse input uses `StandardScaler(with_mean=False)`;
dense input is still centered and scaled. The lower-level functions in
`clustering_tuning` retain their existing `standardize=True` default and now
skip t-SNE unless `compute_tsne=True` is requested. Their `preprocessed_data` result is now sparse
when the input is sparse.

`full_pipeline(visualize=False)` skips t-SNE by default. Set `visualize=True`
to compute coordinates for a plot, or set `compute_tsne=True` to compute
coordinates without plotting. Perplexity, early exaggeration, and
learning-rate options are forwarded on both fixed and tuned paths; fixed
t-SNE is computed only for visualization and is not part of UMAP/HDBSCAN clustering search. `tune_tsne=True` and `tsne_param_grid` now raise a clear error because those values were sampled without being evaluated. Use fixed `tsne_params` for visualization settings.

When t-SNE receives sparse text features, it first applies TruncatedSVD to at
most 50 components. This avoids materializing the full document-vocabulary
matrix, though t-SNE still has document-count-dependent compute and memory costs.

Clustering tuning reports assigned-document `coverage` and excludes candidates below `min_coverage` (default `0.5`) from selection. Set `min_coverage` to a value from 0 to 1 to adjust this eligibility cutoff. Davies-Bouldin and Calinski-Harabasz scores are still calculated on assigned documents only, so compare them alongside coverage.

The legacy `metric=` argument remains an alias for **UMAP only**. Use
`hdbscan_metric=` to select HDBSCAN's metric independently. Changing the old
shared metric or disabling mean centering can change cluster assignments, so
reassess results when migrating an existing analysis.

Explicit `hdbscan_metric="cosine"` still requires a dense distance matrix.
Before fitting UMAP or allocating that matrix, the code checks its float64
size against `max_pairwise_memory_mb` (default **512 MiB**). The limit covers
the distance matrix only; HDBSCAN needs additional working memory. An oversized
fixed request raises `MemoryError`; tuning records an oversized trial in its
error history and continues. Increase the limit explicitly only when memory
allows. The default HDBSCAN search grid uses Euclidean distance; custom grids
remain available through the lower-level tuning functions.

UMAP and HDBSCAN still need memory for neighbor graphs and fitting. Sparse
storage removes the avoidable dense document-vocabulary allocation; it does
not make total memory use independent of dataset size.

## Evaluation Metrics

- **Davies-Bouldin Score**: Measures average similarity between clusters (lower is better)  
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)  
- **Cluster Count**: Number of discovered clusters  
- **Noise Ratio**: Proportion of outlier points  

## Output Formats

### Cluster Summary
DataFrame with columns: Cluster ID, Size, Percentage  

### Keywords DataFrame
DataFrame with columns: cluster, Yake Long, Yake Short, TF, TFIDF, TFDF  

### Topic Analysis
Dictionary containing:
- `cluster_topics`: Mapping of clusters to top topics  
- `topic_designators`: Matching of topics to designators  
- `model`: TopicModeler instance  

## Dependencies

- **Core**: pandas, numpy, scikit-learn  
- **Clustering**: umap-learn, hdbscan  
- **Visualization**: matplotlib, seaborn  
- **NLP**: nltk, gensim, yake  
- **Deep Learning**: transformers, torch  

## Contributing

Contributions are welcome! We encourage you to:

- **Report Issues**: Found a bug or have a feature request? Open an issue on GitHub  
- **Submit Pull Requests**: Improvements to code, documentation, or tests are appreciated  
- **Share Use Cases**: Let us know how you're using tactik  
- **Improve Documentation**: Help us make TACTIK more accessible  

### Development Setup

```bash
# Clone the repository
git clone https://github.com/npsAub/tactik.git
cd tactik

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
unittest discover

# Run linting
flake8 tactik/
black tactik/
```

### Contribution Guidelines

1. Fork the repository and create a feature branch  
2. Write clear, documented code following the existing style  
3. Add tests for new functionality  
4. Update documentation as needed  
5. Submit a pull request with a clear description  

For major changes, please open an issue first to discuss your proposal.

## Citation

If you use TACTIK in your research, please cite:

```bibtex
@software{tactik,
  title={tactik: Text Analysis, Clustering, Tuning, Information and Keyword Extraction},
  author={Niklas P. Schulmeyer and Nicoletta Fala},
  year={2025},
  url={https://github.com/npsAub/tactik}
}
```

## License

MIT License — See `license.txt` for details

## Contact

For questions or support, please open an issue on GitHub or contact [nps0027@auburn.edu].

