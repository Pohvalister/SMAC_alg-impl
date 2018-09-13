import time
import warnings
import solver_base
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.model_selection import RandomizedSearchCV

from sklearn import cluster, datasets, mixture
np.random.seed(0)

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1


datasets = [noisy_circles, noisy_moons, varied, aniso, blobs, no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============

    k_means_base = cluster.KMeans()
    k_means_modify = solver_base.Solver(cluster.KMeans(), solver_base.k_means_params())
    gaussian_base = mixture.GaussianMixture()
    gaussian_modify = solver_base.Solver(mixture.GaussianMixture(), solver_base.gaussian_mixture())
    gaussian_good = mixture.GaussianMixture(n_components=3, covariance_type='full')
    grid_params = solver_base.get_grid_params(solver_base.gaussian_mixture())
    gaussian_grid = RandomizedSearchCV(gaussian_base, grid_params)

    # dbscan1 = cluster.DBSCAN(eps=0.15)
    clustering_algorithms = (
        ('GAUSS-BASE', gaussian_base),
        ('GAUSS-MODIFY', gaussian_modify),
        ('GAUSS-GOOD', gaussian_good),
        ('GAUSS-GRID', gaussian_grid),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                category=UserWarning)
            algorithm = algorithm.fit(X)
            print(name, " score is ", algorithm.score(X))

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()

