import os
import sys

sys.path.append("..")
import requests
import json
import re
import numpy as np
import pandas as pd
from publicsuffixlist import PublicSuffixList
from stringexperiment.char_feature import extract_all_features
from sklearn.externals import joblib
from datetime import datetime
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool


class cluster_process():

    def __init__(self):
        self.AGDPath = "/home/public/2019-01-07-dgarchive_full"
        self.psl = PublicSuffixList()

    def virus_total_test(self, d):
        try:
            url = 'https://www.virustotal.com/vtapi/v2/url/report'
            params = {'apikey': 'f76bdbc3755b5bafd4a18436bebf6a47d0aae6d2b4284f118077aa0dbdbd76a4',
                      'resource': d}
            response = requests.get(url, params=params)
            flag = False
            scan = response.json().get("scans")
            if scan != None:
                for (k, v) in scan.items():
                    if v.get("result") == "malware site" or v.get("result") == "malicious site":
                        flag = True
                        break
            return flag
        except:
            print("{} error check".format(d))
            return False

    #get AGDs samples
    def get_malicious_AGD(self, day):
        # get all AGD
        AGDSet = set()
        with open("../source/dgarchive_full.txt", "r") as f:
            for r in f:
                AGDSet.add(r.strip())
        print("get all AGDs")
        # get all domain
        domains_set = set()
        for n in range(24):
            filepath = os.path.join("../result_data/{}".format(day), "{}{}".format(day, n))
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for r in f:
                        rsplit = r.strip().split(",")
                        domain = rsplit[1]
                        code = int(rsplit[2])
                        if code != 3 and domain in AGDSet:
                            domains_set.add(domain)
        print("AGDs' number:{}".format(len(domains_set)))
        with open("../result_data/{}/{}_AGDs_in_archive".format(day, day), "w") as f:
            f.write("\n".join(domains_set))
        print("virus total check")
        domain_list = list(domains_set)
        flags = []
        pool = Pool(24)
        for d in domain_list:
            flags.append(pool.apply_async(self.virus_total_test, args=(d,)))
        pool.close()
        pool.join()

        result_domain = []
        for i in range(len(flags)):
            if flags[i].get() == True:
                result_domain.append(domain_list[i])
        print("check total:{} ".format(len(result_domain)))
        with open("../result_data/{}/{}_AGDs_in_virustotal".format(day, day), "w") as f:
            f.write("\n".join(result_domain))

    # get ip and domains which it visited
    def visit_domains_by_same_ip(self, day):
        ip_dict = dict()
        for n in range(24):
            filepath = os.path.join("../result_data/{}".format(day), "{}{}".format(day, n))
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for r in f:
                        rsplit = r.strip().split(",")
                        ip = rsplit[0]
                        d = rsplit[1]
                        code = int(rsplit[2])
                        domains_list = ip_dict.get(ip)
                        if domains_list is None:
                            domains_list = ([], [])
                            ip_dict[ip] = domains_list
                        if code == 3:
                            domains_list[1].append(d)
                        else:
                            domains_list[0].append(d)
        with open("../result_data/{}/{}_ip_dict.json".format(day,day), "w") as f:
            f.write(json.dumps(ip_dict))

    # change domains into vectors
    def get_features(self,domain_list, psl):
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

    # xmeans cluster
    def xmeans_cluster(self,domain_features):
        final_centers=None
        final_radiuses=None
        final_clusters=None
        for i in range(5):
            initial_centers = kmeans_plusplus_initializer(domain_features, 2).initialize()
            # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
            max_num = int(len(domain_features) / 2)
            xmeans_instance = xmeans(domain_features, initial_centers, max_num)
            xmeans_instance.process()
            centers = xmeans_instance.get_centers()
            flag=False
            if i==0 or len(centers)>len(final_centers):
                flag=True
            if flag:
                radiuses = []
                cluster_num = 1
                for cluster in xmeans_instance.get_clusters():
                    cluster_num = cluster_num + 1
                    radius_total = 0.0
                    for i in cluster:
                        dist = np.linalg.norm(domain_features[i] - centers[cluster_num - 2])
                        radius_total += dist
                    radiuses.append(radius_total / len(cluster))
                final_centers=xmeans_instance.get_centers()
                final_radiuses=radiuses
                final_clusters=xmeans_instance.get_clusters()

        return final_centers,final_radiuses,final_clusters

    # domain to features
    def domains_map_features(self,day):
        filepath = "../result_data/{}/{}_ip_dict.json".format(day,day)
        domain_set = set()
        with open(filepath, 'r') as f:
            ip_dict = json.loads(f.read())
        for k, v in ip_dict.items():
            for d in v[0]:
                domain_set.add(d)
            for d in v[1]:
                domain_set.add(d)
        domain_list = list(domain_set)
        print('domains number:{}'.format(len(domain_list)))
        psl = PublicSuffixList()
        domain_features = self.get_features(domain_list, psl)
        np.save("../result_data/{}/{}_all_domain_features.npy".format(day,day), domain_features)
        with open("../result_data/{}/{}_all_domain_list.txt".format(day,day), "w") as f:
            f.write('\n'.join(domain_list))

    # cluster
    def get_AGD_cluster(self,day):
        AGDSet=set()
        all_features=np.load("../result_data/{}/{}_all_domain_features.npy".format(day,day))
        with open("../result_data/{}/{}_all_domain_list.txt".format(day,day), "r") as f:
            all_domains=[r.strip() for r in f]
        domain_feature_map=dict()
        for i in range(len(all_domains)):
            domain_feature_map[all_domains[i]]=all_features[i]
        with open("../result_data/{}/{}_AGDs_in_virustotal".format(day,day),"r") as f:
            for r in f:
                AGDSet.add(r.strip())
        with open("../result_data/{}/{}_ip_dict.json".format(day,day),"r") as f:
            ip_dict=json.loads(f.read())
        ip_dga = []
        for k,v in ip_dict.items():

            ip_dga.clear()
            for d in v[0]:
                ip_dga.append(d)
            if len(ip_dga)==0:
                continue
            else:
                ac_features=[domain_feature_map.get(d) for d in ip_dga]
                nx_features=[domain_feature_map.get(d) for d in v[1]]
                ce,r,cl=self.xmeans_cluster(nx_features)
                for i in range(len(ac_features)):
                    d=ip_dga[i]
                    ac_i=ac_features[i]
                    for j in range(len(ce)):
                        if np.linalg.norm(ac_i-ce[j]) < r[j]:
                            print("{}:{}".format(d,[v[1][ii] for ii in cl[j]]))



if __name__ == "__main__":
    days=["20180503","20180504"]
    cp = cluster_process()
    for day in days:
        # cp.get_malicious_AGD(day)
        # cp.visit_domains_by_same_ip(day)
          cp.domains_map_features(day)


def test_cluster(ac_features, centers, radiuses):
    result = []
    for a in range(len(ac_features)):
        af = ac_features[a]
        c = -1
        for i in range(len(centers)):
            dist = np.linalg.norm(af - centers[i])
            if i == 0:
                min_dist = dist
            if dist <= radiuses[i] and dist <= min_dist:
                c = i
                min_dist = dist
        result.append(c)
    return result


def filter_data():
    AGDs = set()
    with open("../result_data/20180427_find_AGD", "r") as f:
        for d in f:
            AGDs.add(d.strip())
    result = []
    with open("../result_data/filter", "r") as f:
        for l in f:
            l_split = l.strip().split(",")
            d = l_split[0]
            if d in AGDs:
                result.append(l.strip())
    with open("../result_data/20180427filter", "w") as f:
        f.write("\n".join(result))


def look_for_malicious_ip_addresses(day="20180427"):
    AGDs = set()
    with open("../result_data/{}_find_AGD".format(day), "r") as f:
        for d in f:
            AGDs.add(d.strip())
    result_set = set()
    for n in range(24):
        filepath = os.path.join("../result_data", "{}{}".format(day, n))
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for r in f:
                    rsplit = r.strip().split(",")
                    d = rsplit[1]
                    answer = rsplit[3]
                    if d in AGDs:
                        result_set.add("{}:{}".format(d, answer))
    with open("{}_ip_addresses".format(day), "w") as f:
        f.write("\n".join(result_set))


def get_related_domains(day="20180427"):
    ip_dict = dict()
    ipv4_pattern = "(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
    with open("../result_data/20180427_ip_addresses", "r") as f:
        for line in f:
            line_split = line.strip().split(":")
            if len(line_split[1]) == 0:
                continue
            else:
                ips = line_split[1].split(";")
                if re.match(ipv4_pattern, ips[0]):
                    for ip in ips:
                        ip_dict[ip] = line_split[0]
                else:
                    continue
    AGDs = set()
    with open("../result_data/{}_find_AGD".format(day), "r") as f:
        for d in f:
            AGDs.add(d.strip())
    result_set = set()
    for n in range(24):
        filepath = os.path.join("../result_data", "{}{}".format(day, n))
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for r in f:
                    rsplit = r.strip().split(",")
                    d = rsplit[1]
                    answer = rsplit[3]
                    if d not in AGDs and len(answer) > 0:
                        for ans in answer.split(";"):
                            if ip_dict.__contains__(ans):
                                result_set.add(r.strip())
    with open("{}_related_domains".format(day), "w") as f:
        f.write("\n".join(result_set))