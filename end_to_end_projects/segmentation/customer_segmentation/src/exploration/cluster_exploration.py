import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ElbowMethod():
    def __init__(self,min_clusters,max_clusters,wcss=None):
        """[summary]

        Args:
            min_clusters ([int]): [Minimum amount of clusters to try for the elbow method]
            max_clusters ([int]): [Maximum amount of clusters to try for the elbow method]
            wcss (Ignore): [Within-Cluster-Sum-of-Squares]. Defaults to None.
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters + 1
        if wcss is None:
            self._wcss = []

    def set_wcss(self,X):
        """[summary]

        Args:
            X ([array_like]):
        """
        self.wcss = []
        for i in range(self.min_clusters, self.max_clusters):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            self.wcss.append(kmeans.inertia_)
    def plot_elbow_method(self):
        if self.wcss == []:
            raise Exception("need to set_wcss")
        labels = [i for i in range(self.min_clusters,self.max_clusters)]
        plt.plot(range(self.min_clusters,self.max_clusters), self.wcss, 'bx-')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.xticks(labels)
        plt.ylabel('WCSS')
        plt.show()