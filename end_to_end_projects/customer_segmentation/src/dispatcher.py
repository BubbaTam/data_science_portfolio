
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, Birch, DBSCAN, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np

kmeans_cluster = 4
Agglomerative_Clusters = None
birch_thresh = None 
birch_clusters = None
MiniBatchKMeans_clusters = None

OPTICS_eps = None 
OPTICS_min_sample = None 

SpectralClustering_clusters = None
GaussianMixture_clusters = None


features = {
    'continuous_feat' : ['age','annual_income_(k$)','spending_score_(1-100)'],
    'nominal_feat' : ['gender'],
    'ordinal_feat' : None,
    'all_feat' : ['gender','age','annual_income_(k$)','spending_score_(1-100)']
}

models = {
    'Kmeans': KMeans(n_clusters=kmeans_cluster, init = 'k-means++'),
    'AgglomerativeClustering' : AgglomerativeClustering(n_clusters=Agglomerative_Clusters),
    'Birch' : Birch(threshold=birch_thresh,n_clusters=birch_clusters),
    'AffinityPropagation' : AffinityPropagation(),
    'DBSCAN' : DBSCAN(eps=0.3,n_jobs=-1),
    #'MiniBatchKMeans' : MiniBatchKMeans(n_cluster=MiniBatchKMeans_clusters),
    'MeanShift' : MeanShift(bandwidth=None),
    'OPTICS' : OPTICS(eps=OPTICS_eps,min_samples=OPTICS_min_sample),
    'SpectralClustering' : SpectralClustering(n_clusters=SpectralClustering_clusters),
    'GaussianMixture' : GaussianMixture(n_components = GaussianMixture_clusters)
}

Hyperparmeter_tuning ={
    'Kmeans' : {
        'n_clusters' : [range(2,11)],
        'init' : 'k-means++',
        'n_init' : 10, #default value 
        'max_iter' : 300, #default
        'tol' : 1e-4, #default  
        'verbose' : 0, #default
        'random_state' : None, # global value of 44
        'copy_x' : True,
        'algorithm' : 'auto'
    },
    'AgglomerativeClustering' : {
        'n_clusters' : 2, #default
        'affinity' : 'euclidean', #default
        'memory' : None, #default
        'connectivity' : None, #default
        'compute_full_tree' : 'auto', #default
        'linkage' : 'ward', # default
        'distance_threshold' : None, #default
        'compute_distances' : None # default
    },
    'BIRCH' : {
        'threshold': 0.5, #default
        'branching_factor' : 50, #default
        'n_clusters' : 3, #default
        'compute_labels' : True, #default
        'copy' : True #default
    },
    'AffinityPropagation' : {
        'damping' : None,
        'max_iter' : None, 
        'convergence_iter' : None, 
        'copy' : None,
        'preference' : None,
        'affinity' : 'euclidean', #default
        'verbose' : False, # default
        'random_state' : None
    },
    'DBSCAN' : {
        'eps' : 0.5, #default
        'min_samples' : 5, #default
        'metric' : 'euclidean', #default
        'metric_params' : None, #default
        'algorithm' : 'auto', #default
        'leaf_size' : 30, #default
        'p' : None, #default
        'n_jobs' : None #default
    },
    'MeanShift' : {
        'bandwidth' : None,
        'seeds' : None, 
        'bin_seeding' : False, 
        'min_bin_freq' : 1, 
        'cluster_all' : True, 
        'n_jobs' : None, 
        'max_iter' : 300
    },
    'OPTICS' : {
        'min_samples' : 5, #default
        'max_eps' : np.inf, #default
        'metric' : 'minkowski', #default
        'p' : 2, #default
        'metric_params' : None, #default
        'cluster_method' : 'xi', #default
        'eps' : None, #default
        'xi' : 0.05, #default
        'predecessor_correction' : True, #default 
        'min_cluster_size' : None, #default
        'algorithm' : 'auto', #default
        'leaf_size' : 30, #default
        'memory' : None, #default
        'n_jobs' : None #default
    },
    'SpectralClustering' : {
        'n_clusters' : 8, #default
        'eigen_solver' : None, #default
        'n_components' : None, #default
        'random_state' : None, #default
        'n_init' : 10, #default
        'gamma' : 1.0, #default
        'affinity' : 'rbf', #default
        'n_neighbors' : 10, #default
        'eigen_tol' : 0.0, #default
        'assign_labels' : 'kmeans', #default
        'degree' : 3, #default
        'coef0' : 1, #default
        'kernel_params' : None, #default 
        'n_jobs' : None, #default
        'verbose' : False #default
    },
    'GaussianMixture' : {
        'n_components' : 1, #default
        'covariance_type' : 'full', #default
        'tol' : 0.001, #default
        'reg_covar' : 1e-06, #default
        'max_iter' : 100, #default
        'n_init' : 1, #default
        'init_params' : 'kmeans', #default
        'weights_init' : None, #default
        'means_init' : None, #default
        'precisions_init' : None, #default
        'random_state' : None, #default
        'warm_start' : False, #default
        'verbose' : 0, #default
        'verbose_interval' : 10 #default
    }
}