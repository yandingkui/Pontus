import os
import sys

sys.path.append("..")
import requests
import json
import numpy as np
from publicsuffixlist import PublicSuffixList
from stringexperiment.char_feature import extract_all_features
from sklearn.externals import joblib
from datetime import datetime


def get_suspicious(year, month, day):
    timestring = "{}{:0>2d}{:0>2d}".format(year, month, day)
    suspicious_domains_set = set()
    if os.path.exists("../result_data/{}domains.txt".format(timestring)):
        with open("../result_data/{}domains.txt".format(timestring), "r") as f:
            for r in f:
                suspicious_domains_set.add(r.strip())
        check_active_domains(suspicious_domains_set, timestring)
    else:
        init_domain_set = set()
        # get all domains
        for hour in range(24):
            file_path = "{}{:0>2d}{:0>2d}{:0>2d}".format(year, month, day, hour)
            if not os.path.exists("../result_data/{}".format(file_path)):
                continue
            with open("../result_data/{}".format(file_path), "r") as f:
                for r in f:
                    domain = r.strip().split(",")[1]
                    init_domain_set.add(domain)
        psl = PublicSuffixList()
        domain_labels = []
        labels_labels = []
        i = 0
        # get labels
        domains_list = list(init_domain_set)
        for d in domains_list:
            s = d[:d.index(psl.publicsuffix(d)) - 1]
            for l in s.split("."):
                if len(l) > 0:
                    domain_labels.append(l)
                    labels_labels.append(i)
            i = i + 1

        features_path = "../result_data/{}_features.npy".format(timestring)
        if os.path.exists(features_path):
            features = np.load(features_path)
        else:
            features = extract_all_features(domain_labels)
            np.save(features_path, features)

        # classifier identifies labels
        clf = joblib.load("../result_data/ac_model.m")
        pred_labels = clf.predict(features)
        domain_index = set()
        for i in range(len(labels_labels)):
            if pred_labels[i] == 1:
                domain_index.add(labels_labels[i])
        # get suspicious domains

        for index in domain_index:
            ps = psl.privatesuffix(domains_list[index])
            if ps is None:
                continue
            suspicious_domains_set.add(ps)

        print("{} domains".format(len(suspicious_domains_set)))

        with open("../result_data/{}domains.txt".format(timestring), "w") as f:
            f.write("\n".join(suspicious_domains_set))
        print("save finish")
        # dgarchive check
        check_active_domains(suspicious_domains_set, timestring)


def lookup_100(domain_list: str, result, index):
    login = ('iie_ac_cn', 'susanirabluffacuityelbow')
    url = 'https://dgarchive.caad.fkie.fraunhofer.de/reverse'
    response = requests.post(url, auth=login, data=domain_list)
    if response.status_code != 200:
        # print("{} finish ,code != 200".format(index))
        return
    else:
        result_map = json.loads(response.content)
        hits_list = result_map.get("hits")
        if len(hits_list) > 0:
            result[index] = hits_list
    # print("{} check finish".format(index))


def get_nearby_AGDs(domain_dict, filename):
    result = dict()

    for k, hits_list in domain_dict.items():
        choice_list=[]
        for hit_domain in hits_list:
            validity = hit_domain.get("validity")
            from_time = datetime.strptime(validity.get("from"), "%Y-%m-%d %H:%M:%S")
            if from_time.year == 2018 and (from_time.month >= 1 and from_time.month <= 6):
                choice_list.append(hit_domain)
        result[k]=choice_list

    with open(filename + ".json", "w") as f:
        f.write(json.dumps(result))


def check_active_domains(domains_set, date):
    result = dict()
    domain_list = list(domains_set)
    for i in range(0, int(len(domains_set) / 100) + 1):
        lookup_100(",".join(domain_list[i * 100:i * 100 + 100]), result, i)
    with open("../result_data/{}labeleddomains.json".format(date), "w") as f:
        f.write(json.dumps(result))

if __name__ == "__main__":
    get_suspicious(2018, 4, 29)