
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, Birch, DBSCAN, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture

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
    'Kmeans': KMeans(n_clusters=kmeans_cluster, init = 'k-means++',random_state=44),
    'AgglomerativeClustering' : AgglomerativeClustering(n_cluster=Agglomerative_Clusters),
    'Birch' : Birch(threshold=birch_thresh,n_cluster=birch_clusters),
    'AffinityPropagation' : AffinityPropagation(),
    'DBSCAN' : DBSCAN(eps=0.3,n_jobs=-1),
    #'MiniBatchKMeans' : MiniBatchKMeans(n_cluster=MiniBatchKMeans_clusters),
    'MeanShift' : MeanShift(bandwidth=None),
    'OPTICS' : OPTICS(eps=OPTICS_eps,min_sample=OPTICS_min_sample),
    'SpectralClustering' : SpectralClustering(n_cluster=SpectralClustering_clusters),
    'GaussianMixture' : GaussianMixture(n_components = GaussianMixture_clusters)
}
