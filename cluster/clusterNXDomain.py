import sys

sys.path.append("..")
import publicsuffixlist
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np


class Cluster():
    #将域名抽象成向量
    def get_features(self, domains: list):
        return extract_all_features(domains)
    #
    def xmeans_model(self, sample):
        amount_initial_centers = 2
        initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
        # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
        max_num = int(len(sample) / 2)
        xmeans_instance = xmeans(sample, initial_centers, max_num)
        xmeans_instance.process()
        # Extract clustering results: clusters and their centers
        clusters = xmeans_instance.get_clusters()
        centers = xmeans_instance.get_centers()
        return clusters, centers

    def get_domains(self):
        """
        get domains from file
        :return: array-like, domains' string
        """
        domains = list()
        with open("../input_data/cluster_sample.txt", "r") as f:
            for row in f:
                domains.append(row.strip())
        return domains

    def put_domains_by_levels(self, domains):
        """
        put domains with same levels together
        :param domains: array-like, domain string
        :return: dict, key is the level of doamins reduced TLD,value is the doamins' index in domain array
        """
        result = dict()
        index = 0
        for d in domains:
            length = len(d[:d.rindex(psl.publicsuffix(d))].split('.'))
            l = result.get(length)
            if not l:
                l = []
            l.append(index)
            result[length] = l
            index = index + 1
        return result

    def sameSld(self, domains):
        sld_map = dict()
        for d in domains:
            pri = MyPublicSuffixList.psl.privatesuffix(d)
            tld = MyPublicSuffixList.psl.publicsuffix(d)
            sld = pri[:pri.index(tld)]
            tldset = sld_map.get(sld)
            if not tldset:
                tldset = set()
            tldset.add(tld)
            sld_map[sld] = tldset
        num = 0
        max_num = 0
        for (k, v) in sld_map.items():
            if len(v) > 1:
                num = num + 1
                if len(v) > max_num:
                    max_num = len(v)
        return max_num, num

    def cluster(self, domains, nx_domain_feature):
        c = Cluster()
        features = [nx_domain_feature.get(d)[0] for d in domains]
        labels = [nx_domain_feature.get(d)[1] for d in domains]
        domains_level = c.put_domains_by_levels(domains)
        domain_cluster = []
        for (k, v) in domains_level.items():
            if len(v) >= 4:
                same_level_domain_feature = [features[i] for i in v]
                clusters, centers = c.xmeans_model(same_level_domain_feature)
                for _i in range(len(clusters)):
                    clusters_v = clusters[_i]
                    cluster_domain = []
                    cluster_dis = []
                    cluster_label = []
                    for ci in clusters_v:
                        cluster_domain.append(domains[v[ci]])
                        cluster_dis.append(np.linalg.norm(same_level_domain_feature[ci] - centers[_i]))
                        cluster_label.append(labels[v[ci]])
                    radius = (max(cluster_dis) + min(cluster_dis)) / 2.0
                    if radius == 0.0:
                        radius = 1.0
                    malicious = 0
                    finallabel = 0
                    for label in cluster_label:
                        if label == 1:
                            malicious = malicious + 1
                    if malicious / len(labels) >= 0.5:
                        finallabel = 1
                    max_num, num = self.sameSld(cluster_domain)
                    domain_cluster.append([cluster_domain, centers[_i], radius, finallabel, max_num, num])
            else:
                for i in range(len(v)):
                    domain_cluster.append([[domains[v[i]]], features[v[i]], 1.0, labels[v[i]], 0, 0])
        return domain_cluster


