import os
import sys

sys.path.append("..")
import requests
import json
import numpy as np
import pandas as pd
from publicsuffixlist import PublicSuffixList
from stringexperiment.char_feature import extract_all_features
from sklearn.externals import joblib
from datetime import datetime
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler


class local_AGD_test():
    def __init__(self):
        self.AGDPath = "/home/public/2019-01-07-dgarchive_full"
        self.psl = PublicSuffixList()

    def one_day_test(self, day: str):
        # get AGD data
        AGDSet = set()
        for filename in os.listdir(self.AGDPath):
            df = pd.read_csv(os.path.join(self.AGDPath, filename), header=None)
            for d in df.iloc[:, 0]:
                AGDSet.add(d)
        # AGDs_dict=dict()
        # for filename in os.listdir(self.AGDPath):
        #     with open(os.path.join(self.AGDPath, filename),"r") as f:
        #         for line in f:
        #             line_split=line.strip().split(",",1)
        #             domain=line_split[0]
        #             other_information=line_split[1]
        #             info_list=AGDs_dict.get(domain)
        #             if info_list is None:
        #                 info_list=[]
        #             info_list.append(other_information)
        #             AGDs_dict[domain]=info_list
        # AGDSet=set(AGDs_dict.keys())
        print("AGDs statistic finish. The number={}".format(len(AGDSet)))
        # get domains
        domains_set = set()
        for n in range(24):
            filepath = os.path.join("../result_data", "{}{}".format(day, n))
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for r in f:
                        rsplit = r.strip().split(",")
                        domain = rsplit[1]
                        code = int(rsplit[2])
                        if code != 3:
                            domains_set.add(domain)
        print("All domains statistic finish. The number = {}".format(len(domains_set)))
        # search
        result = []
        for d in domains_set:
            pri_d = self.psl.privatesuffix(d)
            if d in AGDSet:
                result.append("{}{}".format(d, '*'))
            elif pri_d in AGDSet:
                result.append(d)
        print("search result. The number = {}".format(len(result)))
        with open("../result_data/{}_one_day_temp".format(day), "w") as f:
            f.write("\n".join(result))

    def visit_domains_by_same_ip(self, day):
        AGDSet = set()
        for filename in os.listdir(self.AGDPath):
            df = pd.read_csv(os.path.join(self.AGDPath, filename), header=None)
            for d in df.iloc[:, 0]:
                AGDSet.add(d)
        print("AGDs statistic finish. The number={}".format(len(AGDSet)))
        # get domains
        domains_set = set()
        ip_dict = dict()
        for n in range(24):
            filepath = os.path.join("../result_data", "{}{}".format(day, n))
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for r in f:
                        rsplit = r.strip().split(",")
                        ip = rsplit[0]
                        d = rsplit[1]
                        code = int(rsplit[2])
                        pri_d = self.psl.privatesuffix(d)
                        if d in AGDSet or pri_d in AGDSet:
                            domains_set.add(d)
                        domains_list = ip_dict.get(ip)
                        if domains_list is None:
                            domains_list = ([], [])
                            ip_dict[ip] = domains_list

                        if code == 3:
                            domains_list[1].append(d)
                        else:
                            domains_list[0].append(d)
        print("All domains statistic finish. The number = {}".format(len(domains_set)))

        with open("../result_data/{}_ip_dict.json".format(day), "w") as f:
            f.write(json.dumps(ip_dict))


def get_features(domain_list, psl):
    domain_labels = []
    domain_indexes = []
    index = 0
    for d in domain_list:
        d = d[:d.rindex(psl.publicsuffix(d)) - 1]
        if len(d) == 0:
            continue
        d_labels = d.split(".")
        for l in d_labels:
            domain_labels.append(l)
            domain_indexes.append(index)
        index = index + 1
    label_features = extract_all_features(domain_labels)

    minMax = MinMaxScaler()
    label_features = minMax.fit_transform(label_features)
    zero_array = np.zeros(32)
    domain_features = []
    i = 0
    while i < len(label_features):
        if i == len(label_features) - 1 or domain_indexes[i] != domain_indexes[i + 1]:
            domain_features.append(np.append(zero_array, label_features[i]))
            i = i + 1
        else:
            domain_features.append(np.append(label_features[i], label_features[i + 1]))
            i = i + 2
    return domain_features


def xmeans_cluster(domain_features):
    initial_centers = kmeans_plusplus_initializer(domain_features, 2).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    max_num = int(len(domain_features) / 2)
    xmeans_instance = xmeans(domain_features, initial_centers, max_num)
    xmeans_instance.process()
    radiuses = []
    cluster_num = 1
    centers = xmeans_instance.get_centers()
    for cluster in xmeans_instance.get_clusters():
        cluster_num = cluster_num + 1
        radius_total = 0.0
        for i in cluster:
            dist = np.linalg.norm(domain_features[i] - centers[cluster_num - 2])
            radius_total += dist
        radiuses.append(radius_total / len(cluster))
    return xmeans_instance.get_centers(), radiuses, xmeans_instance.get_clusters()


def test_cluster():
    psl = PublicSuffixList()
    with open("../result_data/20180427_ip_dict.json", "r") as f:
        ip_dict = json.loads(f.read())
    for k, v in ip_dict.items():
        if len(v[1]) > 1000:
            nx = v[1]
            ac = v[0]
            break
    nx_features = get_features(nx, psl)
    centers, radiuses, cluster_indexes = xmeans_cluster(nx_features)
    ac_features = get_features(ac, psl)
    result = []
    for i in range(len(centers)):
        result.append([])
    for a in range(len(ac_features)):
        af = ac_features[a]
        for i in range(len(centers)):
            dist = np.linalg.norm(af - centers[i])
            if dist <= radiuses[i]:
                result[i].append(ac[a])
    print(result)


def domains_map_features(filepath="../result_data/20180427_ip_dict.json"):
    domain_set=set()
    with open(filepath,'r') as f:
        ip_dict=json.loads(f.read())
    for k,v in ip_dict.items():
        for d in v[0]:
            domain_set.add(d)
        for d in v[1]:
            domain_set.add(d)
    domain_list=list(domain_set)
    print('domains number:{}'.format(len(domain_list)))
    psl=PublicSuffixList()
    domain_features=get_features(domain_list,psl)
    np.save("../result_data/all_domain_features.npy",domain_features)
    with open("../result_data/all_domain_list.txt","w") as f:
        f.write('\n'.join(domain_list))

if __name__ == "__main__":
    # lat = local_AGD_test()
    # lat.visit_domains_by_same_ip(day="20180427")
    domains_map_features()