from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np


def xmeans_model(sample):
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    max_num = int(len(sample) / 2)
    xmeans_instance = xmeans(sample, initial_centers, max_num)
    xmeans_instance.process()
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()


    return clusters,centers

def testCluster():
    points=[
        [0,0,0],
        [0,0,1],
        [7,7,7],
        [7,6,7],
        [8,8,8],
        [134,20,89],
        [143,20,87]
    ]
    print(np.var([points[0],points[1]]))
    print(xmeans_model(points))

if __name__=="__main__":
    testCluster()
