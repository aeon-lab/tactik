"""Focused tests for plotting helpers and their input validation."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from visualization import (
    get_cluster_markers,
    get_cluster_palette,
    plot_cluster_comparison,
    plot_clusters,
    plot_clusters_with_annotations,
)


class TestVisualizationHelpers(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_palette_and_markers_have_requested_size(self):
        palette = get_cluster_palette(3)
        markers = get_cluster_markers(12)

        self.assertEqual(len(palette), 3)
        self.assertEqual(len(markers), 12)
        self.assertEqual(markers[0], markers[10])
        self.assertEqual(markers[1], markers[11])

    def test_plot_clusters_includes_outlier_group_and_labels_axes(self):
        embeddings = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        labels = np.array([0, 0, 1, -1])

        figure = plot_clusters(embeddings, labels, title='Test clusters', show=False)

        self.assertEqual(len(figure.axes), 1)
        axes = figure.axes[0]
        self.assertEqual(axes.get_title(), 'Test clusters')
        self.assertEqual(axes.get_xlabel(), 'Dimension 1')
        self.assertEqual(len(axes.collections), 3)

    def test_all_noise_clusters_plot_without_palette_errors(self):
        embeddings = np.array([[0., 0.], [1., 1.], [2., 2.]])
        labels = np.array([-1, -1, -1])

        figure = plot_clusters(embeddings, labels, show=False)
        figure.canvas.draw()

        self.assertEqual(len(figure.axes[0].collections), 1)
        np.testing.assert_allclose(
            figure.axes[0].collections[0].get_facecolors()[0, :3], (0.5, 0.5, 0.5))

    def test_regular_cluster_color_stays_distinct_from_outliers(self):
        embeddings = np.array([[0., 0.], [1., 1.], [2., 2.]])
        labels = np.array([-1, 0, 1])

        figure = plot_clusters(embeddings, labels, show=False)
        figure.canvas.draw()
        gray = np.array([0.5, 0.5, 0.5])
        regular_colors = [collection.get_facecolors()[0, :3]
                          for collection in figure.axes[0].collections[1:]]

        self.assertTrue(all(not np.allclose(color, gray) for color in regular_colors))

    def test_annotated_plot_filters_cluster_outliers(self):
        embeddings = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        clusters = np.array([0, 1, -1, 1])
        annotations = np.array(['a', 'b', 'a', 'b'])

        figure = plot_clusters_with_annotations(
            embeddings, clusters, annotations, filter_outliers=True, show=False)
        plotted_points = sum(len(collection.get_offsets())
                             for collection in figure.axes[0].collections)

        self.assertEqual(plotted_points, 3)

    def test_comparison_plot_creates_two_panels(self):
        embeddings = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        clusters = np.array([0, 0, 1, 1])
        annotations = np.array(['a', 'a', 'b', 'b'])

        figure = plot_cluster_comparison(embeddings, clusters, annotations, show=False)

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(figure.axes[0].get_title(), 'Cluster Assignments')
        self.assertEqual(figure.axes[1].get_title(), 'Annotation Labels')

    def test_comparison_plot_handles_noise_and_all_noise(self):
        embeddings = np.array([[0., 0.], [1., 1.], [2., 2.]])
        annotations = np.array(['a', 'b', 'c'])

        mixed_figure = plot_cluster_comparison(
            embeddings, np.array([-1, 0, 1]), annotations, show=False)
        noise_only_figure = plot_cluster_comparison(
            embeddings, np.array([-1, -1, -1]), annotations, show=False)

        self.assertEqual(len(mixed_figure.axes), 2)
        self.assertEqual(len(noise_only_figure.axes), 2)

    def test_plot_rejects_invalid_shapes_and_nonfinite_coordinates(self):
        with self.assertRaisesRegex(ValueError, 'same length'):
            plot_clusters(np.zeros((3, 2)), np.array([0, 1]), show=False)
        with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
            plot_clusters(np.array([[0., np.nan]]), np.array([0]), show=False)


if __name__ == '__main__':
    unittest.main()
