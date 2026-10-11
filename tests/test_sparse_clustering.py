"""Sparse storage, scaling, and bounded cosine-distance regressions."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, issparse
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import clustering_tuning as ct


class TestSparsePreparation(unittest.TestCase):
    def test_scaling_preserves_sparsity_and_input(self):
        values = np.array([[0., 2., 0.], [1., 0., 4.], [0., 3., 2.]])
        expected = StandardScaler(with_mean=False).fit_transform(values)
        for matrix_type in (csr_matrix, csc_matrix):
            with self.subTest(format=matrix_type.__name__):
                source = matrix_type(values)
                prepared, scaler = ct._prepare_clustering_input(source, True)
                self.assertTrue(issparse(prepared))
                self.assertEqual(prepared.format, 'csr')
                self.assertEqual(prepared.nnz, source.nnz)
                self.assertFalse(scaler.with_mean)
                np.testing.assert_allclose(prepared.toarray(), expected)
                np.testing.assert_array_equal(source.toarray(), values)

    def test_no_scaling_preserves_values_and_copies_input(self):
        source = csr_matrix([[0., 2.], [3., 0.]])
        prepared, scaler = ct._prepare_clustering_input(source, False)
        self.assertIsNone(scaler)
        self.assertEqual((source != prepared).nnz, 0)
        prepared.data[:] = 99
        np.testing.assert_array_equal(source.data, [2., 3.])

    def test_dense_scaling_keeps_centering_and_does_not_mutate_input(self):
        source = np.array([[1., 5.], [3., 9.], [2., 7.]])
        original = source.copy()
        prepared, scaler = ct._prepare_clustering_input(source, True)
        self.assertTrue(scaler.with_mean)
        np.testing.assert_allclose(prepared.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_array_equal(source, original)

    def test_empty_sparse_inputs_are_rejected(self):
        for shape in [(0, 3), (3, 0)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                ct.apply_best_clustering(csr_matrix(shape), {}, compute_tsne=False)


class TestSparsePipelineStorage(unittest.TestCase):
    def test_large_shape_never_materializes_document_vocabulary_matrix(self):
        # 8 GB if accidentally made dense; just one nonzero per row here.
        source = csr_matrix((np.ones(50_000),
                             (np.arange(50_000), np.arange(50_000) % 20_000)),
                            shape=(50_000, 20_000))
        for tune in (False, True):
            for standardize in (False, True):
                with self.subTest(tune=tune, standardize=standardize):
                    reducer = Mock()
                    reducer.fit_transform.return_value = np.zeros((50_000, 2))
                    clusterer = Mock(labels_=np.zeros(50_000, dtype=int))
                    clusterer.fit.return_value = clusterer
                    with patch.object(csr_matrix, 'toarray', side_effect=AssertionError('densified')), \
                            patch.object(ct, 'UMAP', return_value=reducer), \
                            patch.object(ct.hdbscan, 'HDBSCAN', return_value=clusterer), \
                            patch.object(ct, 'pairwise_distances') as distances, \
                            patch.object(ct, 'TSNE') as tsne, \
                            patch.object(ct, 'evaluate_clustering_quality', return_value={
                                'davies_bouldin': 1., 'n_clusters': 2,
                                'coverage': 1.0}):
                        options = dict(standardize=standardize, compute_tsne=False, progress=False)
                        if tune:
                            result = ct.full_clustering_pipeline(
                                source, min_clusters=2, max_iterations=1, **options)
                        else:
                            result = ct.full_clustering_pipeline_fixed_params(source, **options)
                    self.assertTrue(result['success'])
                    self.assertTrue(issparse(result['preprocessed_data']))
                    self.assertEqual(result['preprocessed_data'].shape, source.shape)
                    self.assertEqual(result['preprocessed_data'].nnz, source.nnz)
                    for call in reducer.fit_transform.call_args_list:
                        self.assertIs(call.args[0], result['preprocessed_data'])
                    distances.assert_not_called()
                    tsne.assert_not_called()

    def test_real_sparse_umap_hdbscan_fixed_and_tuned(self):
        dense, _ = make_blobs(n_samples=60, centers=3, n_features=5,
                             cluster_std=0.2, random_state=7)
        source = csr_matrix(dense)
        umap = {'n_neighbors': 8, 'n_components': 2, 'n_epochs': 30,
                'metric': 'euclidean', 'random_state': 42}
        hdbscan = {'min_cluster_size': 3, 'min_samples': 1, 'metric': 'euclidean'}
        with patch.object(ct, 'TSNE') as tsne:
            fixed = ct.full_clustering_pipeline_fixed_params(
                source, umap_params=umap, hdbscan_params=hdbscan,
                standardize=True, compute_tsne=False)
            tuned = ct.full_clustering_pipeline(
                source, umap_param_grid={key: [value] for key, value in umap.items()},
                hdbscan_param_grid={key: [value] for key, value in hdbscan.items()},
                standardize=True, compute_tsne=False, progress=False,
                min_clusters=2, max_iterations=1)
        tsne.assert_not_called()
        for result in (fixed, tuned):
            self.assertTrue(result['success'], result['error_message'])
            self.assertTrue(issparse(result['preprocessed_data']))
            fitted = result['clustering_results']
            self.assertTrue(issparse(fitted['umap_model']._raw_data))
            self.assertEqual(len(fitted['clusters']), 60)
            self.assertIsNone(fitted['tsne_embeddings'])
        np.testing.assert_array_equal(fixed['clustering_results']['clusters'],
                                      tuned['clustering_results']['clusters'])
        np.testing.assert_array_equal(source.toarray(), dense)


class TestCosineMemoryLimit(unittest.TestCase):
    def test_oversized_request_fails_before_umap_or_distances(self):
        params = {'umap': {}, 'hdbscan': {'metric': 'cosine'}, 'tsne': {}}
        with patch.object(ct, 'UMAP') as reducer, \
                patch.object(ct, 'pairwise_distances') as distances:
            with self.assertRaisesRegex(MemoryError, '19073.5 MiB'):
                ct.apply_best_clustering(csr_matrix((50_000, 20_000)), params,
                                         compute_tsne=False)
        reducer.assert_not_called()
        distances.assert_not_called()

    def test_fixed_pipeline_forwards_custom_limit(self):
        with patch.object(ct, 'UMAP') as reducer:
            with self.assertRaises(MemoryError):
                ct.full_clustering_pipeline_fixed_params(
                    csr_matrix(np.eye(10)), hdbscan_params={'metric': 'cosine'},
                    compute_tsne=False, max_pairwise_memory_mb=0.0001)
        reducer.assert_not_called()

    def test_tuner_records_oversized_trial_without_allocating(self):
        with patch.object(ct, 'UMAP') as reducer:
            result = ct.full_clustering_pipeline(
                csr_matrix(np.eye(10)), hdbscan_param_grid={'metric': ['cosine']},
                compute_tsne=False, progress=False, max_iterations=1,
                max_pairwise_memory_mb=0.0001)
        self.assertFalse(result['success'])
        self.assertIn('distance matrix', result['tuning_results']['history'][0]['error'])
        reducer.assert_not_called()

    def test_small_cosine_request_preserves_clusterer_options(self):
        reduced = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
        reducer = Mock()
        reducer.fit_transform.return_value = reduced
        with patch.object(ct, 'UMAP', return_value=reducer), \
                patch.object(ct.hdbscan, 'HDBSCAN') as factory:
            ct._apply_umap_hdbscan(
                np.eye(3), {}, {'metric': 'cosine', 'min_cluster_size': 2,
                               'min_samples': 1, 'cluster_selection_method': 'leaf'},
                max_pairwise_memory_mb=1)
        factory.assert_called_once_with(metric='precomputed', min_cluster_size=2,
                                        min_samples=1, cluster_selection_method='leaf')
        distances = factory.return_value.fit.call_args.args[0]
        self.assertEqual(distances.dtype, np.float64)
        np.testing.assert_allclose(np.diag(distances), 0)

    def test_limit_is_explicitly_configurable(self):
        with self.assertRaises(MemoryError):
            ct._check_pairwise_memory(10, 0.0001)
        ct._check_pairwise_memory(10, 1)
        for limit in [0, -1, float('nan'), float('inf')]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                ct._check_pairwise_memory(10, limit)

if __name__ == '__main__':
    unittest.main()
