import random
import time
import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from umap import UMAP
import hdbscan
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Default parameter grids
DEFAULT_UMAP_GRID = {
    'n_neighbors': [5, 10, 15, 20],
    'min_dist': [0.0, 0.1, 0.2],
    'metric': ['euclidean', 'cosine', 'manhattan'],
    'n_components': [5]  # Fixed default
}

DEFAULT_HDBSCAN_GRID = {
    'min_cluster_size': [10, 15, 20, 25],
    'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
    'cluster_selection_method': ['eom', 'leaf']
}

DEFAULT_TSNE_PARAMS = {
    'perplexity': 30,
    'early_exaggeration': 12,
    'learning_rate': 200,
    'n_components': 2,
    'random_state': 42
}

DEFAULT_TSNE_GRID = {
    'perplexity': [10, 30, 50, 100, 200],
    'early_exaggeration': [10, 20, 30, 50],
    'learning_rate': [100, 200, 500, 1000],
    'n_components': [2]  # Fixed for visualization
}


def evaluate_clustering_quality(clusters, X):
    """
    Evaluates clustering quality using multiple metrics.
    
    This function calculates the Davies-Bouldin score (lower is better) and 
    Calinski-Harabasz score (higher is better) for the given clustering result.
    Noise points (labeled as -1) are filtered out before evaluation.
    
    Args:
        clusters (np.ndarray): Array of cluster labels, where -1 indicates noise points.
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features) on which 
                       clustering was performed.
        
    Returns:
        dict: Dictionary containing:
            - 'davies_bouldin' (float): Davies-Bouldin index (lower is better). 
                                       Returns inf if calculation fails.
            - 'calinski_harabasz' (float): Calinski-Harabasz score (higher is better). 
                                          Returns -inf if calculation fails.
            - 'n_clusters' (int): Number of unique clusters (excluding noise).
            - 'n_noise' (int): Number of noise points.
            - 'noise_ratio' (float): Ratio of noise points to total points.
    
    Raises:
        ValueError: If clusters and X have mismatched dimensions.
    """
    if len(clusters) != len(X):
        raise ValueError(f"Mismatch between clusters length ({len(clusters)}) and X length ({len(X)})")
    
    # Handle empty input
    if len(clusters) == 0:
        return {
            'n_clusters': 0,
            'n_noise': 0,
            'noise_ratio': 0.0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': float('-inf')
        }
    
    metrics = {
        'davies_bouldin': float('inf'),
        'calinski_harabasz': float('-inf'),
        'n_clusters': 0,
        'n_noise': 0,
        'noise_ratio': 0.0
    }
    
    # Filter out noise points (label = -1)
    mask = clusters != -1
    X_filtered = X[mask]
    clusters_filtered = clusters[mask]
    
    # Calculate noise statistics
    metrics['n_noise'] = int(np.sum(~mask))
    metrics['noise_ratio'] = float(metrics['n_noise'] / len(clusters))
    metrics['n_clusters'] = len(np.unique(clusters_filtered))
    
    # Need at least 2 clusters for meaningful metrics
    if len(X_filtered) > 0 and metrics['n_clusters'] > 1:
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, clusters_filtered)
        except Exception as e:
            print(f"Warning: Could not compute Davies-Bouldin score: {e}")
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, clusters_filtered)
        except Exception as e:
            print(f"Warning: Could not compute Calinski-Harabasz score: {e}")
    
    return metrics


def random_parameter_combination(param_grid, random_state=None):
    """
    Randomly select one combination of parameters from a grid.
    
    Args:
        param_grid (dict): Dictionary where keys are parameter names and values are 
                          lists of possible values for that parameter.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        dict: Dictionary with the same keys as param_grid, with randomly selected values.
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    return {key: random.choice(value) for key, value in param_grid.items()}


def _apply_umap_hdbscan(X, umap_params, hdbscan_params, random_state=None):
    """
    Internal function to apply UMAP and HDBSCAN clustering.
    Handles cosine metric special case.
    
    Args:
        X: Input data
        umap_params: UMAP parameters
        hdbscan_params: HDBSCAN parameters
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (umap_embeddings, cluster_labels, hdbscan_model)
    """
    # Add random_state to UMAP params if not already present
    umap_params_copy = umap_params.copy()
    if 'random_state' not in umap_params_copy:
        umap_params_copy['random_state'] = random_state or 42
    
    # Apply UMAP dimensionality reduction
    umap_model = UMAP(**umap_params_copy)
    umap_emb = umap_model.fit_transform(X)
    
    # Handle cosine metric for HDBSCAN on UMAP embeddings
    if hdbscan_params['metric'] == 'cosine':
        # Compute pairwise cosine distances on UMAP embeddings
        distance_matrix = pairwise_distances(umap_emb, metric='cosine')
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_params['min_cluster_size'],
            cluster_selection_method=hdbscan_params['cluster_selection_method'],
            metric='precomputed'
        )
        clusters = hdbscan_model.fit(distance_matrix.astype('float64'))
    else:
        # Apply HDBSCAN directly on UMAP embeddings
        hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params)
        clusters = hdbscan_model.fit(umap_emb)
    
    return umap_emb, clusters.labels_, hdbscan_model


def tune_clustering_hyperparameters(
    X,
    umap_param_grid=None,
    hdbscan_param_grid=None,
    tsne_params=None,
    tune_tsne=False,
    tsne_param_grid=None,
    min_clusters=5,
    time_limit=1800,
    max_iterations=100,
    target_metric='davies_bouldin',
    progress=False,
    random_state=42
):
    """
    Tune hyperparameters for UMAP+HDBSCAN clustering pipeline.
    
    This function performs random search over UMAP and HDBSCAN hyperparameters to find
    the combination that produces the best clustering according to the target metric.
    The pipeline consists of:
    1. UMAP for dimensionality reduction
    2. HDBSCAN for clustering on UMAP embeddings
    3. Optional t-SNE for visualization (can be tuned or use fixed parameters)
    
    Args:
        X (np.ndarray): Input data matrix of shape (n_samples, n_features).
        umap_param_grid (dict, optional): Parameter grid for UMAP. If None, uses DEFAULT_UMAP_GRID.
                                         Keys should be valid UMAP parameters.
        hdbscan_param_grid (dict, optional): Parameter grid for HDBSCAN. If None, uses DEFAULT_HDBSCAN_GRID.
                                            Keys should be valid HDBSCAN parameters.
        tsne_params (dict, optional): Fixed t-SNE parameters to use. If None and tune_tsne=False,
                                     uses DEFAULT_TSNE_PARAMS. Ignored if tune_tsne=True.
        tune_tsne (bool): Whether to tune t-SNE parameters. If False, uses fixed tsne_params.
        tsne_param_grid (dict, optional): Parameter grid for t-SNE tuning. Only used if tune_tsne=True.
                                         If None, uses DEFAULT_TSNE_GRID.
        min_clusters (int): Minimum number of clusters required (excluding noise) for a 
                           configuration to be considered valid.
        time_limit (int): Maximum tuning time in seconds. Tuning stops after this time.
        max_iterations (int): Maximum number of parameter combinations to try.
        target_metric (str): Metric to optimize. Either 'davies_bouldin' (minimize) or 
                            'calinski_harabasz' (maximize).
        progress (bool): Whether to print progress updates every 10 iterations.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Dictionary containing:
            - 'best_params' (dict or None): Best parameters found with keys 'umap', 'hdbscan', 'tsne'.
                                           None if no valid configuration found.
            - 'best_score' (float): Best score achieved for the target metric.
            - 'history' (list): List of dicts, each containing iteration results with keys:
                - 'iteration': iteration number
                - 'params': parameters tried
                - 'metrics': clustering quality metrics
                - 'time_elapsed': cumulative time elapsed
            - 'total_time' (float): Total time taken for tuning.
            - 'iterations_completed' (int): Number of iterations completed.
    
    Raises:
        ValueError: If X is empty, target_metric is invalid, or parameters are malformed.
    """
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Validate input - handle sparse matrices properly
    if X is None:
        raise ValueError("Input data X cannot be None")
    
    try:
        from scipy.sparse import issparse
        if issparse(X):
            if X.shape[0] == 0:
                raise ValueError("Input data X cannot be empty")
        else:
            if len(X) == 0:
                raise ValueError("Input data X cannot be empty")
    except ImportError:
        if len(X) == 0:
            raise ValueError("Input data X cannot be empty")
    
    if target_metric not in ['davies_bouldin', 'calinski_harabasz']:
        raise ValueError("target_metric must be either 'davies_bouldin' or 'calinski_harabasz'")
    
    # Set default parameter grids if not provided
    umap_param_grid = umap_param_grid or DEFAULT_UMAP_GRID.copy()
    hdbscan_param_grid = hdbscan_param_grid or DEFAULT_HDBSCAN_GRID.copy()
    
    # Handle t-SNE parameters
    if tune_tsne:
        tsne_param_grid = tsne_param_grid or DEFAULT_TSNE_GRID.copy()
    else:
        tsne_params = tsne_params or DEFAULT_TSNE_PARAMS.copy()
    
    best_score = float('inf') if target_metric == 'davies_bouldin' else float('-inf')
    best_params = None
    history = []
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            if progress:
                print(f"Time limit of {time_limit}s reached. Stopping tuning.")
            break
        
        try:
            # Randomly sample parameters
            umap_params = random_parameter_combination(umap_param_grid, random_state=random_state + iteration)
            hdbscan_params = random_parameter_combination(hdbscan_param_grid, random_state=random_state + iteration)
            
            # Handle t-SNE parameters
            if tune_tsne:
                current_tsne_params = random_parameter_combination(tsne_param_grid, random_state=random_state + iteration)
                current_tsne_params['random_state'] = random_state + iteration
            else:
                current_tsne_params = tsne_params.copy()
            
            # Apply UMAP and HDBSCAN
            umap_emb, cluster_labels, _ = _apply_umap_hdbscan(X, umap_params, hdbscan_params, random_state=random_state + iteration)
            
            # Evaluate clustering quality on UMAP embeddings (not t-SNE)
            metrics = evaluate_clustering_quality(cluster_labels, umap_emb)
            
            # Determine if this is the best result so far
            current_score = metrics[target_metric]
            valid_clusters = metrics['n_clusters'] >= min_clusters
            
            if valid_clusters:
                is_better = (
                    (target_metric == 'davies_bouldin' and current_score < best_score) or
                    (target_metric == 'calinski_harabasz' and current_score > best_score)
                )
                
                if is_better:
                    best_score = current_score
                    best_params = {
                        'umap': umap_params,
                        'hdbscan': hdbscan_params,
                        'tsne': current_tsne_params
                    }
                    if progress:
                        print(f"New best at iteration {iteration}: {target_metric} = {best_score:.4f}, "
                              f"n_clusters = {metrics['n_clusters']}")
            
            # Store history
            history.append({
                'iteration': iteration,
                'params': {
                    'umap': umap_params,
                    'hdbscan': hdbscan_params,
                    'tsne': current_tsne_params
                },
                'metrics': metrics,
                'time_elapsed': time.time() - start_time
            })
            
            if progress and iteration % 10 == 0:
                print(f"Iteration {iteration}/{max_iterations}: Best {target_metric} = {best_score:.4f}")
        
        except Exception as e:
            if progress:
                print(f"Error at iteration {iteration}: {e}")
            # Store failed iteration in history
            history.append({
                'iteration': iteration,
                'params': None,
                'metrics': None,
                'error': str(e),
                'time_elapsed': time.time() - start_time
            })
            continue
    
    total_time = time.time() - start_time
    
    if progress:
        if best_params is None:
            print(f"No valid clustering found after {len(history)} iterations")
        else:
            print(f"Tuning complete: Best {target_metric} = {best_score:.4f} in {total_time:.2f}s")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'history': history,
        'total_time': total_time,
        'iterations_completed': len(history)
    }


def apply_best_clustering(X, best_params, compute_tsne=True, random_state=42):
    """
    Apply the best found clustering parameters to the data.
    
    This function recreates the full clustering pipeline using the optimal parameters
    found during tuning. It performs UMAP dimensionality reduction, HDBSCAN clustering,
    and optionally t-SNE visualization.
    
    Args:
        X (np.ndarray): Input data matrix of shape (n_samples, n_features). Should be 
                       the same data (or same preprocessing) used during tuning.
        best_params (dict): Best parameters from tuning, containing keys 'umap', 'hdbscan', 
                           and 'tsne' with their respective parameter dictionaries.
        compute_tsne (bool): Whether to compute t-SNE embeddings for visualization. 
                            Set to False to skip t-SNE and improve performance.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Dictionary containing:
            - 'umap_embeddings' (np.ndarray): UMAP reduced features of shape (n_samples, n_components).
            - 'clusters' (np.ndarray): Cluster labels of shape (n_samples,), with -1 for noise points.
            - 'tsne_embeddings' (np.ndarray or None): t-SNE reduced features for visualization.
                                                     None if compute_tsne=False.
            - 'clusterer' (hdbscan.HDBSCAN): Fitted HDBSCAN clusterer object.
            - 'umap_model' (UMAP): Fitted UMAP model.
    
    Raises:
        ValueError: If X is empty or best_params is None/malformed.
        KeyError: If best_params is missing required keys.
    """
    if X is None or len(X) == 0:
        raise ValueError("Input data X cannot be None or empty")
    
    if best_params is None:
        raise ValueError("best_params cannot be None")
    
    required_keys = ['umap', 'hdbscan', 'tsne']
    for key in required_keys:
        if key not in best_params:
            raise KeyError(f"best_params is missing required key: '{key}'")
    
    try:
        # Apply UMAP and HDBSCAN
        umap_params = best_params['umap'].copy()
        hdbscan_params = best_params['hdbscan'].copy()
        
        # Ensure random_state in UMAP params
        if 'random_state' not in umap_params:
            umap_params['random_state'] = random_state
        
        umap_model = UMAP(**umap_params)
        umap_emb = umap_model.fit_transform(X)
        
        _, cluster_labels, hdbscan_model = _apply_umap_hdbscan(X, umap_params, hdbscan_params, random_state=random_state)
        
        # Apply t-SNE on UMAP embeddings for visualization (optional)
        tsne_emb = None
        if compute_tsne:
            try:
                tsne_params = best_params['tsne'].copy()
                # Ensure random_state in tsne params
                if 'random_state' not in tsne_params:
                    tsne_params['random_state'] = random_state
                
                tsne_model = TSNE(**tsne_params)
                tsne_emb = tsne_model.fit_transform(umap_emb)
        
            except Exception as e:
                print(f"Warning: t-SNE computation failed: {e}")
                import traceback
                traceback.print_exc()
                tsne_emb = None
        
        return {
            'umap_embeddings': umap_emb,
            'clusters': cluster_labels,
            'tsne_embeddings': tsne_emb,
            'clusterer': hdbscan_model,
            'umap_model': umap_model
        }
    
    except Exception as e:
        raise RuntimeError(f"Failed to apply clustering with best parameters: {e}")


def full_clustering_pipeline(
    X,
    standardize=True,
    min_clusters=5,
    time_limit=1800,
    max_iterations=100,
    target_metric='davies_bouldin',
    progress=True,
    tsne_params=None,
    tune_tsne=False,
    compute_tsne=True,
    random_state=42
):
    """
    Complete clustering pipeline from raw data to final clusters.
    
    This is a high-level function that orchestrates the entire clustering workflow:
    1. Optional data standardization
    2. Hyperparameter tuning via random search
    3. Application of best parameters to obtain final clustering
    
    The pipeline uses UMAP for dimensionality reduction and HDBSCAN for clustering.
    t-SNE can optionally be used for 2D visualization.
    
    Args:
        X (np.ndarray): Input data matrix of shape (n_samples, n_features).
        standardize (bool): Whether to standardize the data (zero mean, unit variance) 
                           before clustering. Recommended for features on different scales.
        min_clusters (int): Minimum number of clusters required (excluding noise) for a 
                           configuration to be considered valid during tuning.
        time_limit (int): Maximum tuning time in seconds.
        max_iterations (int): Maximum number of parameter combinations to try during tuning.
        target_metric (str): Metric to optimize. Either 'davies_bouldin' (minimize) or 
                            'calinski_harabasz' (maximize).
        progress (bool): Whether to print progress updates during tuning.
        tsne_params (dict, optional): Fixed t-SNE parameters. If None and tune_tsne=False,
                                     uses DEFAULT_TSNE_PARAMS.
        tune_tsne (bool): Whether to tune t-SNE parameters during hyperparameter search.
        compute_tsne (bool): Whether to compute final t-SNE embeddings. Set to False to 
                            skip t-SNE and improve performance.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Dictionary containing:
            - 'tuning_results' (dict): Complete tuning results including best_params, 
                                      best_score, and history.
            - 'clustering_results' (dict or None): Final clustering results including 
                                                  embeddings and cluster labels. 
                                                  None if no valid parameters found.
            - 'preprocessed_data' (np.ndarray): The preprocessed input data (standardized if requested).
            - 'scaler' (StandardScaler or None): The fitted scaler if standardization was applied.
            - 'success' (bool): Whether clustering was successful.
            - 'error_message' (str or None): Error message if clustering failed.
    
    Raises:
        ValueError: If X is empty or invalid.
    """
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Validate input - handle sparse matrices properly
    if X is None:
        raise ValueError("Input data X cannot be None")
    
    try:
        from scipy.sparse import issparse
        if issparse(X):
            if X.shape[0] == 0:
                raise ValueError("Input data X cannot be empty")
            # Convert sparse to dense
            X = X.toarray()
        else:
            if len(X) == 0:
                raise ValueError("Input data X cannot be empty")
    except ImportError:
        if len(X) == 0:
            raise ValueError("Input data X cannot be empty")
    
    result = {
        'tuning_results': None,
        'clustering_results': None,
        'preprocessed_data': None,
        'scaler': None,
        'success': False,
        'error_message': None
    }
    
    try:
        # Standardize data if requested
        X_processed = X.copy()
        scaler = None
        
        if standardize:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            if progress:
                print("Data standardized")
        
        result['preprocessed_data'] = X_processed
        result['scaler'] = scaler
        
        # Tune hyperparameters
        if progress:
            print("Starting hyperparameter tuning...")
        
        tuning_results = tune_clustering_hyperparameters(
            X=X_processed,
            min_clusters=min_clusters,
            time_limit=time_limit,
            max_iterations=max_iterations,
            target_metric=target_metric,
            progress=progress,
            tsne_params=tsne_params,
            tune_tsne=tune_tsne,
            random_state=random_state
        )
        
        result['tuning_results'] = tuning_results
        
        # Apply best parameters if found
        if tuning_results['best_params'] is not None:
            if progress:
                print("Applying best parameters to data...")
            
            clustering_results = apply_best_clustering(
                X_processed, 
                tuning_results['best_params'],
                compute_tsne=compute_tsne,
                random_state=random_state
            )
            
            result['clustering_results'] = clustering_results
            result['success'] = True
            
            if progress:
                n_clusters = len(np.unique(clustering_results['clusters'][clustering_results['clusters'] != -1]))
                n_noise = np.sum(clustering_results['clusters'] == -1)
                print(f"Clustering complete: {n_clusters} clusters found, {n_noise} noise points")
        else:
            result['error_message'] = (
                f"No valid clustering configuration found after {tuning_results['iterations_completed']} iterations. "
                f"Try adjusting min_clusters, increasing time_limit/max_iterations, or modifying parameter grids."
            )
            if progress:
                print(f"Warning: {result['error_message']}")
    
    except Exception as e:
        result['error_message'] = f"Pipeline failed with error: {str(e)}"
        if progress:
            print(f"Error: {result['error_message']}")
        raise
    
    return result


def full_clustering_pipeline_fixed_params(
    X,
    umap_params=None,
    hdbscan_params=None,
    tsne_params=None,
    standardize=True,
    progress=False,
    compute_tsne=True,
    random_state=42
):
    """
    Complete clustering pipeline with fixed parameters (no tuning).
    
    This function runs the clustering pipeline without hyperparameter tuning,
    using provided or default parameters. Useful for applying known good parameters
    or for quick testing.
    
    Args:
        X (np.ndarray): Input data matrix of shape (n_samples, n_features).
        umap_params (dict, optional): UMAP parameters. If None, uses sensible defaults.
        hdbscan_params (dict, optional): HDBSCAN parameters. If None, uses sensible defaults.
        tsne_params (dict, optional): t-SNE parameters. If None, uses DEFAULT_TSNE_PARAMS.
        standardize (bool): Whether to standardize the data before clustering.
        progress (bool): Whether to print progress messages.
        compute_tsne (bool): Whether to compute t-SNE embeddings for visualization.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Dictionary containing:
            - 'clustering_results' (dict): Clustering results including embeddings and labels.
            - 'preprocessed_data' (np.ndarray): The preprocessed input data.
            - 'scaler' (StandardScaler or None): The fitted scaler if standardization was applied.
            - 'success' (bool): Whether clustering was successful.
            - 'error_message' (str or None): Error message if clustering failed.
            - 'params_used' (dict): The parameters that were used.
    
    Raises:
        ValueError: If X is empty or invalid.
    """
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Validate input - handle sparse matrices properly
    if X is None:
        raise ValueError("Input data X cannot be None")
    
    # Check if sparse matrix
    try:
        from scipy.sparse import issparse
        if issparse(X):
            if X.shape[0] == 0:
                raise ValueError("Input data X cannot be empty")
        else:
            if len(X) == 0:
                raise ValueError("Input data X cannot be empty")
    except ImportError:
        # If scipy not available, assume dense array
        if len(X) == 0:
            raise ValueError("Input data X cannot be empty")
    
    # Set default parameters if not provided
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.0,
            'metric': 'cosine',
            'n_components': 5,
            'random_state': random_state
        }
    
    if hdbscan_params is None:
        hdbscan_params = {
            'min_cluster_size': 10,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        }
    
    if tsne_params is None:
        tsne_params = DEFAULT_TSNE_PARAMS.copy()
        tsne_params['random_state'] = random_state
    
    result = {
        'clustering_results': None,
        'preprocessed_data': None,
        'scaler': None,
        'success': False,
        'error_message': None,
        'params_used': {
            'umap': umap_params,
            'hdbscan': hdbscan_params,
            'tsne': tsne_params
        }
    }
    
    try:
        # Convert sparse to dense if needed and copy
        try:
            from scipy.sparse import issparse
            if issparse(X):
                X_processed = X.toarray()
            else:
                X_processed = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        except ImportError:
            X_processed = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        
        scaler = None
        
        if standardize:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            if progress:
                print("Data standardized")
        
        result['preprocessed_data'] = X_processed
        result['scaler'] = scaler
        
        # Create best_params structure for apply_best_clustering
        best_params = {
            'umap': umap_params,
            'hdbscan': hdbscan_params,
            'tsne': tsne_params
        }
        
        if progress:
            print("Applying clustering with fixed parameters...")
        
        clustering_results = apply_best_clustering(
            X_processed,
            best_params,
            compute_tsne=compute_tsne,
            random_state=random_state
        )
        
        result['clustering_results'] = clustering_results
        result['success'] = True
        
        if progress:
            n_clusters = len(np.unique(clustering_results['clusters'][clustering_results['clusters'] != -1]))
            n_noise = np.sum(clustering_results['clusters'] == -1)
            print(f"Clustering complete: {n_clusters} clusters found, {n_noise} noise points")
    
    except Exception as e:
        result['error_message'] = f"Pipeline failed with error: {str(e)}"
        if progress:
            print(f"Error: {result['error_message']}")
        raise
    
    return result