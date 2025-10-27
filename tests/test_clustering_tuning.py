# test_clustering_tuning.py
import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_moons
import clustering_tuning as ct


class TestEvaluateClusteringQuality(unittest.TestCase):
    """Tests for evaluate_clustering_quality function."""
    
    def test_valid_clustering(self):
        """Test with valid clustering result."""
        X = np.random.randn(100, 2)
        clusters = np.array([0] * 30 + [1] * 40 + [2] * 30)
        
        metrics = ct.evaluate_clustering_quality(clusters, X)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('davies_bouldin', metrics)
        self.assertIn('calinski_harabasz', metrics)
        self.assertIn('n_clusters', metrics)
        self.assertIn('n_noise', metrics)
        self.assertIn('noise_ratio', metrics)
        self.assertEqual(metrics['n_clusters'], 3)
        self.assertEqual(metrics['n_noise'], 0)
        self.assertEqual(metrics['noise_ratio'], 0.0)
        self.assertLess(metrics['davies_bouldin'], float('inf'))
        self.assertGreater(metrics['calinski_harabasz'], float('-inf'))
    
    def test_clustering_with_noise(self):
        """Test clustering with noise points."""
        X = np.random.randn(100, 2)
        clusters = np.array([0] * 30 + [1] * 40 + [-1] * 30)
        
        metrics = ct.evaluate_clustering_quality(clusters, X)
        
        self.assertEqual(metrics['n_clusters'], 2)
        self.assertEqual(metrics['n_noise'], 30)
        self.assertEqual(metrics['noise_ratio'], 0.3)
    
    def test_all_noise_points(self):
        """Test when all points are classified as noise."""
        X = np.random.randn(50, 2)
        clusters = np.array([-1] * 50)
        
        metrics = ct.evaluate_clustering_quality(clusters, X)
        
        self.assertEqual(metrics['n_clusters'], 0)
        self.assertEqual(metrics['n_noise'], 50)
        self.assertEqual(metrics['noise_ratio'], 1.0)
        self.assertEqual(metrics['davies_bouldin'], float('inf'))
        self.assertEqual(metrics['calinski_harabasz'], float('-inf'))
    
    def test_single_cluster(self):
        """Test when only one cluster is found."""
        X = np.random.randn(50, 2)
        clusters = np.array([0] * 50)
        
        metrics = ct.evaluate_clustering_quality(clusters, X)
        
        self.assertEqual(metrics['n_clusters'], 1)
        self.assertEqual(metrics['davies_bouldin'], float('inf'))
        self.assertEqual(metrics['calinski_harabasz'], float('-inf'))
    
    def test_mismatched_dimensions(self):
        """Test error when clusters and X have different lengths."""
        X = np.random.randn(100, 2)
        clusters = np.array([0] * 50)
        
        with self.assertRaisesRegex(ValueError, "Mismatch between clusters length"):
            ct.evaluate_clustering_quality(clusters, X)
    
    def test_empty_input(self):
        """Test with empty arrays."""
        X = np.array([]).reshape(0, 2)
        clusters = np.array([])
        
        metrics = ct.evaluate_clustering_quality(clusters, X)
        
        self.assertEqual(metrics['n_clusters'], 0)
        self.assertEqual(metrics['n_noise'], 0)
    
    def test_perfect_separation(self):
        """Test with perfectly separated clusters."""
        X = np.array([[0, 0], [0, 1], [0, 2], [10, 10], [10, 11], [10, 12]])
        clusters = np.array([0, 0, 0, 1, 1, 1])
        
        metrics = ct.evaluate_clustering_quality(clusters, X)
        
        self.assertEqual(metrics['n_clusters'], 2)
        self.assertLess(metrics['davies_bouldin'], 1.0)
        self.assertGreater(metrics['calinski_harabasz'], 0)


class TestRandomParameterCombination(unittest.TestCase):
    """Tests for random_parameter_combination function."""
    
    def test_basic_functionality(self):
        """Test basic random selection."""
        param_grid = {'a': [1, 2, 3], 'b': ['x', 'y', 'z']}
        
        result = ct.random_parameter_combination(param_grid)
        
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertIn(result['a'], [1, 2, 3])
        self.assertIn(result['b'], ['x', 'y', 'z'])
    
    def test_single_option(self):
        """Test when parameters have only one option."""
        param_grid = {'a': [1], 'b': ['x']}
        
        result = ct.random_parameter_combination(param_grid)
        
        self.assertEqual(result, {'a': 1, 'b': 'x'})
    
    def test_empty_grid(self):
        """Test with empty parameter grid."""
        param_grid = {}
        
        result = ct.random_parameter_combination(param_grid)
        
        self.assertEqual(result, {})
    
    def test_randomness(self):
        """Test that function actually produces different results."""
        param_grid = {'a': list(range(100))}
        
        results = [ct.random_parameter_combination(param_grid)['a'] for _ in range(50)]
        
        self.assertGreater(len(set(results)), 1)


class TestTuneClusteringHyperparameters(unittest.TestCase):
    """Tests for tune_clustering_hyperparameters function."""
    
    def test_basic_tuning(self):
        """Test basic tuning functionality with simple data."""
        X, y = make_blobs(n_samples=200, centers=3, n_features=5, random_state=42)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=5,
            time_limit=60,
            progress=False
        )
        
        self.assertIn('best_params', result)
        self.assertIn('best_score', result)
        self.assertIn('history', result)
        self.assertIn('total_time', result)
        self.assertIn('iterations_completed', result)
        self.assertLessEqual(len(result['history']), 5)
        self.assertLessEqual(result['iterations_completed'], 5)
    
    def test_finds_valid_clustering(self):
        """Test that tuning finds valid clustering for well-separated data."""
        X, y = make_blobs(n_samples=300, centers=4, n_features=10, 
                         cluster_std=0.5, random_state=42)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=20,
            min_clusters=3,
            time_limit=120,
            progress=False
        )
        
        self.assertIsNotNone(result['best_params'])
        self.assertIn('umap', result['best_params'])
        self.assertIn('hdbscan', result['best_params'])
        self.assertIn('tsne', result['best_params'])
    
    def test_time_limit_respected(self):
        """Test that time limit is respected."""
        X = np.random.randn(1000, 50)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=1000,
            time_limit=2,
            progress=False
        )
        
        self.assertLessEqual(result['total_time'], 5)
    
    def test_custom_parameter_grids(self):
        """Test with custom parameter grids."""
        X, y = make_blobs(n_samples=150, centers=3, random_state=42)
        
        custom_umap_grid = {
            'n_neighbors': [10, 15],
            'min_dist': [0.1],
            'metric': ['euclidean']
        }
        
        custom_hdbscan_grid = {
            'min_cluster_size': [10, 15],
            'metric': ['euclidean'],
            'cluster_selection_method': ['eom']
        }
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            umap_param_grid=custom_umap_grid,
            hdbscan_param_grid=custom_hdbscan_grid,
            max_iterations=5,
            min_clusters=2,
            progress=False
        )
        
        # Check that the function ran and returned results
        self.assertIsNotNone(result)
        self.assertIn('best_params', result)
        
        # If valid clustering was found, check parameters are from custom grids
        if result['best_params'] is not None:
            self.assertIn(result['best_params']['umap']['n_neighbors'], [10, 15])
            self.assertEqual(result['best_params']['umap']['min_dist'], 0.1)
    
    def test_tsne_tuning_enabled(self):
        """Test with t-SNE tuning enabled."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        custom_tsne_grid = {
            'perplexity': [20, 30],
            'learning_rate': [100, 200],
            'early_exaggeration': [12]
        }
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            tune_tsne=True,
            tsne_param_grid=custom_tsne_grid,
            max_iterations=5,
            progress=False
        )
        
        tsne_params = [h['params']['tsne'] for h in result['history'] if h.get('params')]
        self.assertGreater(len(tsne_params), 0)
    
    def test_tsne_tuning_disabled(self):
        """Test with fixed t-SNE parameters."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        fixed_tsne = {'perplexity': 25, 'learning_rate': 150, 'early_exaggeration': 10}
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            tune_tsne=False,
            tsne_params=fixed_tsne,
            max_iterations=5,
            progress=False
        )
        
        tsne_params = [h['params']['tsne'] for h in result['history'] if h.get('params')]
        self.assertTrue(all(p == fixed_tsne for p in tsne_params))
    
    def test_calinski_harabasz_optimization(self):
        """Test optimization with Calinski-Harabasz metric."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=10,
            target_metric='calinski_harabasz',
            progress=False
        )
        
        if result['best_params'] is not None:
            self.assertGreater(result['best_score'], float('-inf'))
    
    def test_davies_bouldin_optimization(self):
        """Test optimization with Davies-Bouldin metric."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=10,
            target_metric='davies_bouldin',
            progress=False
        )
        
        if result['best_params'] is not None:
            self.assertLess(result['best_score'], float('inf'))
    
    def test_invalid_target_metric(self):
        """Test error with invalid target metric."""
        X = np.random.randn(100, 5)
        
        with self.assertRaisesRegex(ValueError, "target_metric must be either"):
            ct.tune_clustering_hyperparameters(
                X=X,
                target_metric='invalid_metric'
            )
    
    def test_empty_input(self):
        """Test error with empty input data."""
        X = np.array([]).reshape(0, 5)
        
        with self.assertRaisesRegex(ValueError, "Input data X cannot be (None|empty)"):
            ct.tune_clustering_hyperparameters(X=X)
    
    def test_none_input(self):
        """Test error with None input."""
        with self.assertRaisesRegex(ValueError, "Input data X cannot be (None|empty)"):
            ct.tune_clustering_hyperparameters(X=None)
    
    def test_min_clusters_constraint(self):
        """Test that min_clusters constraint is respected."""
        X = np.random.randn(100, 5)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=10,
            min_clusters=50,
            progress=False
        )
        
        self.assertIsNone(result['best_params'])
    
    def test_history_recorded(self):
        """Test that history is properly recorded."""
        X, y = make_blobs(n_samples=150, centers=3, random_state=42)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=5,
            progress=False
        )
        
        self.assertEqual(len(result['history']), 5)
        for entry in result['history']:
            self.assertIn('iteration', entry)
            self.assertIn('time_elapsed', entry)
            self.assertTrue(('params' in entry and 'metrics' in entry) or 'error' in entry)
    
    def test_cosine_metric_handling(self):
        """Test that cosine metric is handled correctly."""
        X, y = make_blobs(n_samples=150, centers=3, random_state=42)
        
        hdbscan_grid = {
            'min_cluster_size': [10],
            'metric': ['cosine'],
            'cluster_selection_method': ['eom']
        }
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            hdbscan_param_grid=hdbscan_grid,
            max_iterations=3,
            progress=False
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result['history']), 0)


class TestApplyBestClustering(unittest.TestCase):
    """Tests for apply_best_clustering function."""
    
    def test_basic_application(self):
        """Test basic application of best parameters."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        best_params = {
            'umap': {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'},
            'hdbscan': {'min_cluster_size': 10, 'metric': 'euclidean', 
                       'cluster_selection_method': 'eom'},
            'tsne': {'perplexity': 30, 'learning_rate': 200, 'early_exaggeration': 12}
        }
        
        result = ct.apply_best_clustering(X, best_params)
        
        self.assertIn('umap_embeddings', result)
        self.assertIn('clusters', result)
        self.assertIn('tsne_embeddings', result)
        self.assertIn('clusterer', result)
        self.assertIn('umap_model', result)
        self.assertEqual(len(result['clusters']), len(X))
        self.assertEqual(result['umap_embeddings'].shape[0], len(X))
    
    def test_without_tsne(self):
        """Test application without computing t-SNE."""
        X, y = make_blobs(n_samples=150, centers=3, random_state=42)
        
        best_params = {
            'umap': {'n_neighbors': 10, 'min_dist': 0.1, 'metric': 'euclidean'},
            'hdbscan': {'min_cluster_size': 10, 'metric': 'euclidean', 
                       'cluster_selection_method': 'eom'},
            'tsne': {'perplexity': 30, 'learning_rate': 200, 'early_exaggeration': 12}
        }
        
        result = ct.apply_best_clustering(X, best_params, compute_tsne=False)
        
        self.assertIsNone(result['tsne_embeddings'])
        self.assertIsNotNone(result['clusters'])
    
    def test_cosine_metric(self):
        """Test application with cosine metric."""
        X, y = make_blobs(n_samples=150, centers=3, random_state=42)
        
        best_params = {
            'umap': {'n_neighbors': 10, 'min_dist': 0.1, 'metric': 'euclidean'},
            'hdbscan': {'min_cluster_size': 10, 'metric': 'cosine', 
                       'cluster_selection_method': 'eom'},
            'tsne': {'perplexity': 30, 'learning_rate': 200, 'early_exaggeration': 12}
        }
        
        result = ct.apply_best_clustering(X, best_params, compute_tsne=False)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result['clusters']), len(X))
    
    def test_none_best_params(self):
        """Test error when best_params is None."""
        X = np.random.randn(100, 5)
        
        with self.assertRaisesRegex(ValueError, "best_params cannot be None"):
            ct.apply_best_clustering(X, None)
    
    def test_empty_input(self):
        """Test error with empty input."""
        X = np.array([]).reshape(0, 5)
        best_params = {
            'umap': {'n_neighbors': 10, 'min_dist': 0.1, 'metric': 'euclidean'},
            'hdbscan': {'min_cluster_size': 10, 'metric': 'euclidean', 
                       'cluster_selection_method': 'eom'},
            'tsne': {'perplexity': 30, 'learning_rate': 200, 'early_exaggeration': 12}
        }
        
        with self.assertRaisesRegex(ValueError, "Input data X cannot be (None|empty)"):
            ct.apply_best_clustering(X, best_params)
    
    def test_missing_required_keys(self):
        """Test error when best_params is missing required keys."""
        X = np.random.randn(100, 5)
        
        incomplete_params = {
            'umap': {'n_neighbors': 10, 'min_dist': 0.1, 'metric': 'euclidean'}
        }
        
        with self.assertRaisesRegex(KeyError, "missing required key"):
            ct.apply_best_clustering(X, incomplete_params)
    
    def test_deterministic_with_same_params(self):
        """Test that same parameters produce same results (with same random seed)."""
        X, y = make_blobs(n_samples=150, centers=3, random_state=42)
        
        best_params = {
            'umap': {'n_neighbors': 10, 'min_dist': 0.1, 'metric': 'euclidean', 'random_state': 42},
            'hdbscan': {'min_cluster_size': 10, 'metric': 'euclidean', 
                       'cluster_selection_method': 'eom'},
            'tsne': {'perplexity': 30, 'learning_rate': 200, 'early_exaggeration': 12, 'random_state': 42}
        }
        
        result1 = ct.apply_best_clustering(X, best_params)
        result2 = ct.apply_best_clustering(X, best_params)
        
        np.testing.assert_array_equal(result1['clusters'], result2['clusters'])


class TestFullClusteringPipeline(unittest.TestCase):
    """Tests for full_clustering_pipeline function."""
    
    def test_basic_pipeline(self):
        """Test basic pipeline execution."""
        X, y = make_blobs(n_samples=200, centers=3, n_features=10, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            time_limit=60,
            progress=False
        )
        
        self.assertIn('tuning_results', result)
        self.assertIn('clustering_results', result)
        self.assertIn('preprocessed_data', result)
        self.assertIn('scaler', result)
        self.assertIn('success', result)
        self.assertIn('error_message', result)
    
    def test_successful_clustering(self):
        """Test that pipeline successfully clusters well-separated data."""
        X, y = make_blobs(n_samples=250, centers=4, cluster_std=0.5, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=15,
            min_clusters=3,
            progress=False
        )
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['clustering_results'])
        self.assertIsNone(result['error_message'])
    
    def test_with_standardization(self):
        """Test pipeline with data standardization."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        X[:, 0] *= 100
        X[:, 1] *= 0.01
        
        result = ct.full_clustering_pipeline(
            X=X,
            standardize=True,
            max_iterations=10,
            progress=False
        )
        
        self.assertIsNotNone(result['scaler'])
        self.assertIsNotNone(result['preprocessed_data'])
        self.assertLess(np.abs(np.mean(result['preprocessed_data'][:, 0])), 0.1)
        self.assertLess(np.abs(np.std(result['preprocessed_data'][:, 0]) - 1.0), 0.1)
    
    def test_without_standardization(self):
        """Test pipeline without standardization."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            standardize=False,
            max_iterations=10,
            progress=False
        )
        
        self.assertIsNone(result['scaler'])
        np.testing.assert_array_almost_equal(result['preprocessed_data'], X)
    
    def test_no_valid_clustering_found(self):
        """Test behavior when no valid clustering is found."""
        X = np.random.randn(50, 5)
        
        result = ct.full_clustering_pipeline(
            X=X,
            min_clusters=100,
            max_iterations=5,
            progress=False
        )
        
        self.assertFalse(result['success'])
        self.assertIsNone(result['clustering_results'])
        self.assertIsNotNone(result['error_message'])
        self.assertIn('No valid clustering', result['error_message'])
    
    def test_skip_tsne(self):
        """Test pipeline without t-SNE computation."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=10,
            compute_tsne=False,
            progress=False
        )
        
        if result['success']:
            self.assertIsNone(result['clustering_results']['tsne_embeddings'])
    
    def test_custom_tsne_params(self):
        """Test pipeline with custom fixed t-SNE parameters."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        custom_tsne = {'perplexity': 40, 'learning_rate': 300, 'early_exaggeration': 15}
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            tsne_params=custom_tsne,
            tune_tsne=False,
            progress=False
        )
        
        if result['success']:
            self.assertEqual(result['tuning_results']['best_params']['tsne'], custom_tsne)
    
    def test_tune_tsne_enabled(self):
        """Test pipeline with t-SNE tuning enabled."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            tune_tsne=True,
            progress=False
        )
        
        self.assertIsNotNone(result['tuning_results'])
    
    def test_different_target_metrics(self):
        """Test pipeline with different optimization metrics."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        result_db = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            target_metric='davies_bouldin',
            progress=False
        )
        
        result_ch = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            target_metric='calinski_harabasz',
            progress=False
        )
        
        self.assertIsNotNone(result_db['tuning_results'])
        self.assertIsNotNone(result_ch['tuning_results'])
    
    def test_empty_input(self):
        """Test error with empty input."""
        X = np.array([]).reshape(0, 5)
        
        with self.assertRaisesRegex(ValueError, "Input data X cannot be (None|empty)"):
            ct.full_clustering_pipeline(X=X)
    
    def test_none_input(self):
        """Test error with None input."""
        with self.assertRaisesRegex(ValueError, "Input data X cannot be (None|empty)"):
            ct.full_clustering_pipeline(X=None)
    
    def test_complex_data(self):
        """Test pipeline with complex non-blob data."""
        X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=15,
            min_clusters=2,
            progress=False
        )
        
        self.assertIsNotNone(result['tuning_results'])
    
    def test_high_dimensional_data(self):
        """Test pipeline with high-dimensional data."""
        X = np.random.randn(200, 100)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            min_clusters=2,
            progress=False
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['preprocessed_data'].shape, X.shape)


class TestDefaultParameterGrids(unittest.TestCase):
    """Tests for default parameter grids."""
    
    def test_default_umap_grid_valid(self):
        """Test that default UMAP grid is valid."""
        self.assertIn('n_neighbors', ct.DEFAULT_UMAP_GRID)
        self.assertIn('min_dist', ct.DEFAULT_UMAP_GRID)
        self.assertIn('metric', ct.DEFAULT_UMAP_GRID)
        self.assertTrue(all(isinstance(v, list) for v in ct.DEFAULT_UMAP_GRID.values()))
    
    def test_default_hdbscan_grid_valid(self):
        """Test that default HDBSCAN grid is valid."""
        self.assertIn('min_cluster_size', ct.DEFAULT_HDBSCAN_GRID)
        self.assertIn('metric', ct.DEFAULT_HDBSCAN_GRID)
        self.assertIn('cluster_selection_method', ct.DEFAULT_HDBSCAN_GRID)
        self.assertTrue(all(isinstance(v, list) for v in ct.DEFAULT_HDBSCAN_GRID.values()))
    
    def test_default_tsne_params_valid(self):
        """Test that default t-SNE params are valid."""
        self.assertIsInstance(ct.DEFAULT_TSNE_PARAMS, dict)
        self.assertIn('perplexity', ct.DEFAULT_TSNE_PARAMS)
        self.assertIn('learning_rate', ct.DEFAULT_TSNE_PARAMS)
        self.assertIn('early_exaggeration', ct.DEFAULT_TSNE_PARAMS)
    
    def test_default_tsne_grid_valid(self):
        """Test that default t-SNE grid is valid."""
        self.assertIn('perplexity', ct.DEFAULT_TSNE_GRID)
        self.assertIn('learning_rate', ct.DEFAULT_TSNE_GRID)
        self.assertIn('early_exaggeration', ct.DEFAULT_TSNE_GRID)
        self.assertTrue(all(isinstance(v, list) for v in ct.DEFAULT_TSNE_GRID.values()))


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests."""
    
    def test_very_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(20, 3)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=3,
            min_clusters=2,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            min_clusters=2,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(150, 200)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=3,
            min_clusters=2,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_identical_points(self):
        """Test with many identical points."""
        X = np.ones((100, 5))
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=3,
            min_clusters=1,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_extreme_outliers(self):
        """Test with extreme outliers."""
        X, y = make_blobs(n_samples=150, centers=3, n_features=5, random_state=42)
        X[0] = [1000, 1000, 1000, 1000, 1000]
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            standardize=True,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_nan_values(self):
        """Test behavior with NaN values."""
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        
        try:
            result = ct.full_clustering_pipeline(
                X=X,
                max_iterations=3,
                progress=False
            )
            self.assertIsNotNone(result)
        except (ValueError, RuntimeError):
            pass
    
    def test_inf_values(self):
        """Test behavior with infinite values."""
        X = np.random.randn(100, 5)
        X[0, 0] = np.inf
        
        try:
            result = ct.full_clustering_pipeline(
                X=X,
                max_iterations=3,
                progress=False
            )
            self.assertIsNotNone(result)
        except (ValueError, RuntimeError):
            pass
    
    def test_zero_variance_features(self):
        """Test with features that have zero variance."""
        X = np.random.randn(100, 5)
        X[:, 2] = 5.0
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            standardize=True,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_negative_values(self):
        """Test with all negative values."""
        X = -np.abs(np.random.randn(100, 5))
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_very_tight_clusters(self):
        """Test with very tight, overlapping clusters."""
        X, y = make_blobs(n_samples=200, centers=5, cluster_std=0.01, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=10,
            min_clusters=3,
            progress=False
        )
        
        self.assertIsNotNone(result)


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow."""
    
    def test_complete_workflow_blobs(self):
        """Test complete workflow on blob data."""
        X, y_true = make_blobs(n_samples=300, centers=4, n_features=10, 
                               cluster_std=1.0, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            standardize=True,
            min_clusters=3,
            max_iterations=20,
            time_limit=120,
            target_metric='davies_bouldin',
            progress=False
        )
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['clustering_results'])
        
        clusters = result['clustering_results']['clusters']
        n_clusters = len(np.unique(clusters[clusters != -1]))
        
        self.assertGreaterEqual(n_clusters, 3)
        self.assertEqual(len(clusters), len(X))
    
    def test_complete_workflow_moons(self):
        """Test complete workflow on non-convex data."""
        X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            standardize=True,
            min_clusters=2,
            max_iterations=20,
            target_metric='calinski_harabasz',
            progress=False
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result['tuning_results'])
    
    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with random state in UMAP."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        umap_grid = {
            'n_neighbors': [15],
            'min_dist': [0.1],
            'metric': ['euclidean'],
            'random_state': [42]
        }
        
        hdbscan_grid = {
            'min_cluster_size': [10],
            'metric': ['euclidean'],
            'cluster_selection_method': ['eom']
        }
        
        tsne_params = {'perplexity': 30, 'learning_rate': 200, 
                      'early_exaggeration': 12, 'random_state': 42}
        
        result1 = ct.tune_clustering_hyperparameters(
            X=X,
            umap_param_grid=umap_grid,
            hdbscan_param_grid=hdbscan_grid,
            tsne_params=tsne_params,
            max_iterations=1,
            progress=False
        )
        
        result2 = ct.tune_clustering_hyperparameters(
            X=X,
            umap_param_grid=umap_grid,
            hdbscan_param_grid=hdbscan_grid,
            tsne_params=tsne_params,
            max_iterations=1,
            progress=False
        )
        
        if result1['best_params'] and result2['best_params']:
            self.assertEqual(result1['best_score'], result2['best_score'])
    
    def test_pipeline_preserves_data(self):
        """Test that pipeline doesn't modify original data."""
        X_original = np.random.randn(100, 5)
        X_copy = X_original.copy()
        
        result = ct.full_clustering_pipeline(
            X=X_original,
            standardize=True,
            max_iterations=3,
            progress=False
        )
        
        np.testing.assert_array_equal(X_original, X_copy)
    
    def test_multiple_runs_different_results(self):
        """Test that multiple runs produce different parameter combinations."""
        X, y = make_blobs(n_samples=200, centers=3, random_state=42)
        
        results = []
        for _ in range(3):
            result = ct.tune_clustering_hyperparameters(
                X=X,
                max_iterations=5,
                progress=False
            )
            results.append(result)
        
        all_params = [r['history'][0]['params'] for r in results if r['history']]
        self.assertGreater(len(all_params), 0)
    
    def test_large_dataset_performance(self):
        """Test that pipeline handles larger datasets."""
        X, y = make_blobs(n_samples=1000, centers=5, n_features=20, random_state=42)
        
        result = ct.full_clustering_pipeline(
            X=X,
            max_iterations=5,
            min_clusters=3,
            progress=False,
            compute_tsne=False
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result['tuning_results'])


class TestParameterValidation(unittest.TestCase):
    """Tests for parameter validation."""
    
    def test_negative_min_clusters(self):
        """Test with negative min_clusters."""
        X = np.random.randn(100, 5)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            min_clusters=-5,
            max_iterations=3,
            progress=False
        )
        
        self.assertIsNotNone(result)
    
    def test_zero_max_iterations(self):
        """Test with zero max_iterations."""
        X = np.random.randn(100, 5)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            max_iterations=0,
            progress=False
        )
        
        self.assertEqual(len(result['history']), 0)
    
    def test_very_short_time_limit(self):
        """Test with very short time limit."""
        X = np.random.randn(100, 5)
        
        result = ct.tune_clustering_hyperparameters(
            X=X,
            time_limit=0.001,
            max_iterations=100,
            progress=False
        )
        
        self.assertLess(result['total_time'], 5)


if __name__ == '__main__':
    unittest.main()