import sys
import json
import os

sys.path.append("..")
from multiprocessing import Pool, Manager
import subprocess
import traceback
from Levenshtein import distance
import pandas as pd
from util.MyJsonEncoder import MyJsonEncoder
import random
from util.MyPublicSuffixList import MyPublicSuffixList
from sklearn.model_selection import GridSearchCV
import char_feature

import data_clean.myFeatures as myFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


###############################
def get_l2_singlefile(time_file, fname):
    result = set()
    with open("/home/yandingkui/dga_detection/result_data/" + time_file + "/" + fname, "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        if len(v[1]) == 0:
            continue
        else:
            for i in v[1]:
                for j in v[0]:
                    dis = distance(i, j)
                    if len(i.strip().split(".")) == 2 and (dis == 1 or dis == 2) and i != j:
                        result.add(i)
    print("{} finish, size={}".format(fname, len(result)))
    return result


# get correct domain names which have 2 levels
def get_correct_level2(time_files):
    nx_l2_list = []
    pool = Pool(processes=24)
    for time_file in time_files:
        print(time_file)
        files = os.listdir("/home/yandingkui/dga_detection/result_data/" + time_file)
        for fname in files:
            if not fname.endswith("json"):
                continue
            res = pool.apply_async(get_l2_singlefile, args=(time_file, fname,))
            nx_l2_list.append(res)
    pool.close()
    pool.join()
    result = set()
    for res in nx_l2_list:
        for r in res.get():
            result.add(r)
    with open("/home/yandingkui/dga_detection/result_data/NXD_l2", "w") as f:
        f.write("\n".join(result) + "\n")
###############################

###############################
def run(fname: str, d, total, time_file):
    try:
        if (not fname.endswith("json")) or (not fname.startswith("ac_nx_")):
            return
        print(fname)
        nx_set = set()
        with open("/home/yandingkui/dga_detection/result_data/" + time_file + "/" + fname, "r") as f:
            map = json.loads(f.read())
        for k, v in map.items():
            for i in v[1]:
                nx_set.add(i)
        print(len(nx_set))
        for i in nx_set:
            two = i[i.index(".") + 1:]
            if not d.get(two):
                try:
                    p = subprocess.Popen("host " + two, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ns_r = p.stdout.read()
                    if "NXDOMAIN" not in str(ns_r):
                        print(i)
                    d[two] = True
                except:
                    continue
            else:
                continue
    except:
        print(traceback.print_exc())


# get correct domain names which level is 3
def get_correct_level3(time_file):
    isCheck = Manager().dict()
    total_dict = Manager().dict()
    files = os.listdir("/home/yandingkui/dga_detection/result_data/" + time_file)

    pool = Pool(processes=24)

    for fname in files:
        pool.apply_async(run, args=(fname, isCheck, total_dict, time_file,))
    pool.close()
    pool.join()


###############################

###############################
# get AGDs from dgarchive
def get_dgarchive_data(root_dir="/home/yandingkui/dga_detection/input_data/dgarchive/full_dgarchive/"):
    files = os.listdir(root_dir)

    dga_list = list()
    for filename in files:
        df = pd.read_csv(root_dir + filename, header=None, error_bad_lines=False)
        if df.shape[0] > 365:
            category = filename.strip().split(".")[0]
            a_set = set()
            a_list = []
            for i in df.iloc[:, 0]:
                si = i.split(".")[0]
                if si in a_set:
                    continue
                else:
                    a_list.append(i)
                    a_set.add(si)
            dga_list.append([a_list, category])

    dga_list = sorted(dga_list, key=lambda x: len(x[0]))

    unique = set()
    result = dict()

    for item_list in dga_list:

        number = len(item_list[0])
        if number < 498:
            continue
        elif number >= 498 and number < 1000:
            end_list = []
            for domain in item_list[0]:
                si = domain.strip().split(".")[0]
                if si in unique:
                    continue
                else:
                    end_list.append(domain)
                    unique.add(si)
                    if len(end_list) == 500:
                        break
            result[item_list[1]] = end_list
        else:
            end_list = []
            sample_list = random.sample(item_list[0], 1000)
            for domain in sample_list:
                si = domain.strip().split(".")[0]
                if si in unique:
                    continue
                else:
                    end_list.append(domain)
                    unique.add(si)
                    if len(end_list) == 500:
                        break
            result[item_list[1]] = end_list
    for k, v in result.items():
        print("{}:{}".format(k, len(v)))
    with open("/home/yandingkui/dga_detection/result_data/AGDs", "w") as f:
        f.write(json.dumps(result, cls=MyJsonEncoder))


###############################

def NXDomain_dataset(root_dir="/home/yandingkui/dga_detection/result_data/"):
    result = []
    with open(root_dir + "correct_level2", "r") as f:
        for r in f:
            result.append(r.strip())
    old_3 = []

    with open(root_dir + "correct_level3", "r") as f:
        for r in f:
            old_3.append(r.strip())
    new_3 = random.sample(old_3, 29500 - len(result) + 1)
    for r in new_3:
        result.append(r)
    with open(root_dir + "benign_nx", "w") as f:
        f.write("\n".join(result))


def get_NXDomain_dataset(root_dir="/home/yandingkui/dga_detection/result_data/"):
    df = pd.DataFrame(columns=('domains', 'labels'))
    with open(root_dir + "benign_nx", "r") as f:
        for r in f:
            s = pd.Series({'domains': r.strip(), 'labels': 0})
            df = df.append(s, ignore_index=True)
    with open(root_dir + "AGDs.json", "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        for r in v:
            s = pd.Series({'domains': r.strip(), 'labels': 1})
            df = df.append(s, ignore_index=True)
    df.to_csv(root_dir + "nx_dataset.csv")


###############################
def suffix_1m_single(time_file, fname, judge_set):
    result = set()
    with open("/home/yandingkui/dga_detection/result_data/" + time_file + "/" + fname, "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        if len(v[0]) == 0:
            continue
        else:
            for i in v[0]:
                if MyPublicSuffixList.psl.privatesuffix(i) in judge_set:
                    result.add(i)
    print("{} finish, size={}".format(fname, len(result)))
    return result


def get_benign_activedomain(time_files=["20180427", "20180428", "20180429", "20180430"]):
    top1m = pd.read_csv("/home/yandingkui/dga_detection/referenceFiles/top-1m.csv", )
    judge_set = set()
    for r in top1m.iloc[100001:, 1]:
        judge_set.add(r.strip())

    nx_l2_list = []
    pool = Pool(processes=24)
    for time_file in time_files:
        print(time_file)
        files = os.listdir("/home/yandingkui/dga_detection/result_data/" + time_file)
        for fname in files:
            if not fname.endswith("json"):
                continue
            res = pool.apply_async(suffix_1m_single, args=(time_file, fname, judge_set,))
            nx_l2_list.append(res)
    pool.close()
    pool.join()
    result = set()
    for res in nx_l2_list:
        for r in res.get():
            result.add(r)
    with open("/home/yandingkui/dga_detection/result_data/AC", "w") as f:
        f.write("\n".join(result) + "\n")


def get_active_dataset():
    ol2 = []
    ol3 = []
    with open("/home/yandingkui/dga_detection/result_data/AC", "r") as f:
        for r in f:
            level = len(r.strip().split("."))
            if level == 2:
                ol2.append(r.strip())
            elif level == 3:
                ol3.append(r.strip())
    nl2 = random.sample(ol2, 20000)
    nl3 = random.sample(ol3, 9500)
    result = nl2 + nl3

    with open("/home/yandingkui/dga_detection/result_data/benign_ac", "w") as f:
        f.write("\n".join(result))


def get_active_domain_dataset(root_dir="/home/yandingkui/dga_detection/result_data/"):
    df = pd.DataFrame(columns=('domains', 'labels'))
    with open(root_dir + "benign_ac", "r") as f:
        for r in f:
            s = pd.Series({'domains': r.strip(), 'labels': 0})
            df = df.append(s, ignore_index=True)
    with open(root_dir + "AGDs.json", "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        for r in v:
            s = pd.Series({'domains': r.strip(), 'labels': 1})
            df = df.append(s, ignore_index=True)
    df.to_csv(root_dir + "ac_dataset.csv")


###############################

def get_word_list_domain(root_dir="/home/yandingkui/dga_detection/input_data/dgarchive/full_dgarchive/"):
    files = ["gozi_dga.csv", "suppobox_dga.csv", "matsnu_dga.csv"]
    result = dict()
    for filename in files:
        df = pd.read_csv(root_dir + filename, header=None, error_bad_lines=False)
        category = filename.strip().split(".")[0]
        a_set = set()
        a_list = []
        for i in df.iloc[:, 0]:
            si = i.split(".")[0]
            if si in a_set:
                continue
            else:
                a_list.append(i)
                a_set.add(si)
            result[category] = a_list
            if len(a_list) == 10000:
                break
    with open("/home/yandingkui/dga_detection/result_data/wordlist_AGDs.json", "w") as f:
        f.write(json.dumps(result, cls=MyJsonEncoder))


def get_wordlist_active_dataset():
    ol2 = []
    ol3 = []
    with open("/home/yandingkui/dga_detection/result_data/AC", "r") as f:
        for r in f:
            level = len(r.strip().split("."))
            if level == 2:
                ol2.append(r.strip())
            elif level == 3:
                ol3.append(r.strip())
    nl2 = random.sample(ol2, 20000)
    nl3 = random.sample(ol3, 10000)
    result = nl2 + nl3

    with open("/home/yandingkui/dga_detection/result_data/benign_word_list_ac", "w") as f:
        f.write("\n".join(result))


def wordlist_dataset(root_dir="/home/yandingkui/dga_detection/result_data/"):
    df = pd.DataFrame(columns=('domains', 'labels'))
    with open(root_dir + "benign_word_list_ac", "r") as f:
        for r in f:
            s = pd.Series({'domains': r.strip(), 'labels': 0})
            df = df.append(s, ignore_index=True)
    with open(root_dir + "wordlist_AGDs.json", "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        for r in v:
            s = pd.Series({'domains': r.strip(), 'labels': 1})
            df = df.append(s, ignore_index=True)
    df.to_csv(root_dir + "wordlist_dataset.csv")


################################################################################

def get_active_domain_dataset(root_dir="/home/yandingkui/dga_detection/result_data/"):
    df = pd.DataFrame(columns=('domains', 'labels'))
    with open(root_dir + "benign_ac", "r") as f:
        for r in f:
            s = pd.Series({'domains': r.strip(), 'labels': 0})
            df = df.append(s, ignore_index=True)
    with open(root_dir + "AGDs.json", "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        for r in v:
            s = pd.Series({'domains': r.strip(), 'labels': 1})
            df = df.append(s, ignore_index=True)
    df.to_csv(root_dir + "ac_dataset.csv")


def split_AGDs(root_dir="/home/yandingkui/dga_detection/result_data/"):
    index = range(500)
    pred_index = random.sample(index, 100)
    # print("{}:{}".format(len(pred_L),pred_L))
    with open(root_dir + "AGDs.json", "r") as f:
        map = json.loads(f.read())
    result = dict()
    for k, v in map.items():
        train_list = []
        pred_list = []
        for i in range(500):
            if i in pred_index:
                pred_list.append(v[i])
            else:
                train_list.append(v[i])
        result[k] = (train_list, pred_list)
    with open("/home/yandingkui/dga_detection/result_data/split_AGDs", "w") as f:
        f.write(json.dumps(result, cls=MyJsonEncoder))


def split_ac_nx(root_dir="/home/yandingkui/dga_detection/result_data/"):
    benign_ac = []
    benign_nx = []
    with open(root_dir + "benign_ac", "r") as f:
        for r in f:
            benign_ac.append(r.strip())
    with open(root_dir + "benign_nx", "r") as f:
        for r in f:
            benign_nx.append(r.strip())

    index = range(29500)
    pred_index = random.sample(index, 23600)
    benign_ac_train = []
    benign_nx_train = []
    benign_ac_pred = []
    benign_nx_pred = []

    for i in range(29500):
        if i in pred_index:
            benign_ac_train.append(benign_ac[i])
            benign_nx_train.append(benign_nx[i])
        else:
            benign_ac_pred.append(benign_ac[i])
            benign_nx_pred.append(benign_nx[i])

    ac_result = dict()
    ac_result["train"] = benign_ac_train
    ac_result["pred"] = benign_ac_pred
    nx_result = dict()
    nx_result["train"] = benign_nx_train
    nx_result["pred"] = benign_nx_pred
    with open(root_dir + "split_benign_ac.json", "w") as f:
        f.write(json.dumps(ac_result))
    with open(root_dir + "split_benign_nx.json", "w") as f:
        f.write(json.dumps(nx_result))


def FANCI_expirement_process(root_dir="/home/yandingkui/dga_detection/result_data/", m_file="split_AGDs",
                             benign_file="split_benign_ac.json", n=755, m=28, c='gini'):
    print("FANCI:{}".format(benign_file))
    with open(root_dir + m_file, "r") as f:
        malicious_data = json.loads(f.read())

    with open(root_dir + benign_file, "r") as f:
        benign_data = json.loads(f.read())
    train_domains = []
    train_labels = []
    pred_domains = []
    pred_labels = []
    for k, v in malicious_data.items():
        for d in v[0]:
            train_domains.append(d)
            train_labels.append(1)
        for d in v[1]:
            pred_domains.append(d)
            pred_labels.append(1)

    # print(train_domains)
    # print(pred_domains)

    for d in benign_data.get("train"):
        train_domains.append(d)
        train_labels.append(0)
    for d in benign_data.get("pred"):
        pred_domains.append(d)
        pred_labels.append(0)

    # print(len(train_domains))
    # print(len(pred_domains))

    train_features = myFeatures.extract_all_features(train_domains)

    index = list(range(len(train_domains)))
    random.shuffle(index)

    real_train_features = []
    real_train_labels = []
    for i in index:
        real_train_features.append(train_features[i])
        real_train_labels.append(train_labels[i])

    clf = RandomForestClassifier(n_estimators=n, max_features=m, criterion=c)
    # clf.fit(real_train_features, real_train_labels)
    # print("features")
    # n_es_list = range(750, 850, 5)
    # max_fea_list = range(10, 30, 2)
    # tuned_parameters = [{'n_estimators': n_es_list, 'random_state': [0],
    #                     'max_features': max_fea_list, 'criterion': ["gini", "entropy"]}]

    # clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
    #                  scoring='accuracy', n_jobs=30)

    clf.fit(real_train_features, real_train_labels)
    # print("best_params:")
    # print(clf.best_params_)

    # print(clf.feature_importances_)
    pred_features = myFeatures.extract_all_features(pred_domains)
    predict_result = []
    # predict_result = clf.predict(pred_features)
    predict_result_p = clf.predict_proba(pred_features)
    for p in predict_result_p:
        predict_result.append(p[1])
    # print(len(pred_labels))
    # print(len(predict_result))
    # print(predict_result)
    ROC_dict = dict()
    ROC_dict["real"] = pred_labels
    ROC_dict["predict"] = predict_result
    with open("/home/yandingkui/dga_detection/result_data/" + benign_file[benign_file.index("_") + 1:benign_file.rindex(
            ".")] + "FANCI_ROC.json", "w") as f:
        f.write(json.dumps(ROC_dict))
    # print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}" \
    #      .format(accuracy_score(pred_labels, predict_result), \
    #              recall_score(pred_labels, predict_result), \
    #              precision_score(pred_labels, predict_result), \
    #              f1_score(pred_labels, predict_result)))

def takeSecond(elem):
    return elem[1]

def MY_expirement_process(root_dir="/home/yandingkui/dga_detection/result_data/", m_file="split_AGDs",
                          benign_file="split_benign_ac.json", n=815, m=10, c='entropy'):
    with open(root_dir + m_file, "r") as f:
        malicious_data = json.loads(f.read())

    with open(root_dir + benign_file, "r") as f:
        benign_data = json.loads(f.read())

    train_domains = []
    train_labels = []
    pred_domains = []
    pred_labels = []
    for k, v in malicious_data.items():
        for d in v[0]:
            train_domains.append(d.split(".")[0])
            train_labels.append(1)
        for d in v[1]:
            pred_domains.append(d)
            pred_labels.append(1)

    for d in benign_data.get("train"):
        train_domains.append(d.split(".")[0])
        train_labels.append(0)
    for d in benign_data.get("pred"):
        pred_domains.append(d)
        pred_labels.append(0)

    train_features = char_feature.extract_all_features(train_domains)

    index = list(range(len(train_domains)))
    random.shuffle(index)

    real_train_features = []
    real_train_labels = []
    for i in index:
        real_train_features.append(train_features[i])
        real_train_labels.append(train_labels[i])

    # clf = RandomForestClassifier(n_estimators=800, random_state=0)
    # {'criterion': 'entropy', 'max_features': 14, 'n_estimators': 820, 'random_state': 0}
    clf = RandomForestClassifier(n_estimators=n, max_features=m, criterion=c, random_state=0)
    # print("features")
    # n_es_list=range(750,850,5)
    # max_fea_list=range(10,30,2)
    # tuned_parameters = [{'n_estimators':n_es_list , 'random_state': [0],'max_features': max_fea_list,'criterion':["gini","entropy"]}]

    # clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,scoring='accuracy',n_jobs=30)

    clf.fit(real_train_features, real_train_labels)
    # print("best_params:")
    # print(clf.best_params_)
    print("Pontus:feature_importance_")
    im=clf.feature_importances_
    feature_items=[]
    for i in range(len(im)):
        feature_items.append((i+1,im[i]))
    feature_items.sort(key=takeSecond,reverse=True)
    print(feature_items)

    # print("train")
    # predict_result = []
    #
    # l1_domains = []
    # l2_domains = []
    # l1_domains_lables = []
    # l2_domains_lables = []
    # for i in range(len(pred_domains)):
    #     d = pred_domains[i].strip()
    #     publicsuffix = MyPublicSuffixList.psl.publicsuffix(d)
    #     dp = d[:d.rindex(publicsuffix) - 1]
    #     d_array = dp.split(".")
    #     if len(d_array) == 1:
    #         l1_domains.append(d_array[0])
    #         l1_domains_lables.append(pred_labels[i])
    #     elif len(d_array) == 2:
    #         l2_domains.append(d_array)
    #         l2_domains_lables.append(pred_labels[i])
    #     else:
    #         print("error for {} {} {}".format(d, publicsuffix, d_array))
    # l1_features = char_feature.extract_all_features(l1_domains)
    # l2_features = char_feature.extract_all_features_for_2(l2_domains)
    # pred_r1_p = clf.predict_proba(l1_features)
    # # print(pred_r1)
    # pred_r1 = []
    # for p in pred_r1_p:
    #     pred_r1.append(p[1])
    # # print(pred_r1)
    # pred_r2 = []
    # for item in l2_features:
    #     rrr = clf.predict_proba(item)
    #     # print(rrr)
    #     if rrr[0][1] > rrr[1][1]:
    #         pred_r2.append(rrr[0][1])
    #     else:
    #         pred_r2.append(rrr[1][1])
    #     # if rrr[0] == 0 and rrr[1] == 0:
    #     # pred_r2.append(0)
    #     # else:
    #     # pred_r2.append(1)
    #
    # pred_labels = l1_domains_lables + l2_domains_lables
    # for l in pred_r1:
    #     predict_result.append(l)
    # for l in pred_r2:
    #     predict_result.append(l)
    # # predict_result=pred_r1+pred_r2
    #
    # # print("predict")
    # # print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}" \
    # #      .format(accuracy_score(pred_labels, predict_result), \
    # #              recall_score(pred_labels, predict_result), \
    # #              precision_score(pred_labels, predict_result), \
    # #              f1_score(pred_labels, predict_result)))
    # print(len(pred_labels))
    # print(len(predict_result))
    # ROC_dict = dict()
    # ROC_dict["real"] = pred_labels
    # ROC_dict["predict"] = predict_result
    # with open("/home/yandingkui/dga_detection/result_data/" + benign_file[benign_file.index("_") + 1:benign_file.rindex(
    #         ".")] + "MY_ROC.json", "w") as f:
    #     f.write(json.dumps(ROC_dict))


#################################

def split_wordlist_AGDs(root_dir="/home/yandingkui/dga_detection/result_data/"):
    index = range(10000)
    pred_index = random.sample(index, 2000)
    # print("{}:{}".format(len(pred_L),pred_L))
    with open(root_dir + "wordlist_AGDs.json", "r") as f:
        map = json.loads(f.read())
    result = dict()
    for k, v in map.items():
        train_list = []
        pred_list = []
        for i in range(10000):
            if i in pred_index:
                pred_list.append(v[i])
            else:
                train_list.append(v[i])
        result[k] = (train_list, pred_list)
    with open("/home/yandingkui/dga_detection/result_data/split_wordlist_AGDs.json", "w") as f:
        f.write(json.dumps(result, cls=MyJsonEncoder))


def split_wordlist_benign(root_dir="/home/yandingkui/dga_detection/result_data/"):
    benign_ac = []

    with open(root_dir + "benign_word_list_ac", "r") as f:
        for r in f:
            benign_ac.append(r.strip())

    index = range(30000)
    pred_index = random.sample(index, 24000)
    benign_ac_train = []

    benign_ac_pred = []

    for i in range(30000):
        if i in pred_index:
            benign_ac_train.append(benign_ac[i])

        else:
            benign_ac_pred.append(benign_ac[i])

    ac_result = dict()
    ac_result["train"] = benign_ac_train
    ac_result["pred"] = benign_ac_pred

    with open(root_dir + "split_benign_wordlist_ac.json", "w") as f:
        f.write(json.dumps(ac_result))


def handle_one_day(day):
    try:
        ac_set = set()
        nx_set = set()
        files = os.listdir("../result_data/" + day)
        for f in files:
            if f.endswith(".json"):
                with open("../result_data/" + day + "/" + f, "r") as F:
                    map = json.loads(F.read())
                    for k, v in map.items():
                        for i in v[0]:
                            ac_set.add(i.strip())
                        for i in v[1]:
                            nx_set.add(i.strip())
        result = dict()
        result["ac"] = list(ac_set)
        result["nx"] = list(nx_set)
        with open("../result_data/" + day + "/all_domain.json", "w") as F:
            F.write(json.dumps(result))
        print(day + "   finish")
    except:
        print(traceback.print_exc())


def get_All_domain_all_day(days=["20180427", "20180428", "20180429", "20180430"]):
    pool = Pool(processes=4)
    for day in days:
        pool.apply_async(func=handle_one_day, args=(day,))
    pool.close()
    pool.join()


def real_world(root_dir="/home/yandingkui/dga_detection/result_data/", m_file="split_AGDs",
               benign_file="split_benign_nx.json"):
    with open(root_dir + m_file, "r") as f:
        malicious_data = json.loads(f.read())

    with open(root_dir + benign_file, "r") as f:
        benign_data = json.loads(f.read())
    train_domains = []
    train_labels = []

    for k, v in malicious_data.items():
        for d in v[0]:
            train_domains.append(d.split(".")[0])
            train_labels.append(1)
        for d in v[1]:
            train_domains.append(d.split(".")[0])
            train_labels.append(1)

    for d in benign_data.get("train"):
        train_domains.append(d.split(".")[0])
        train_labels.append(0)

    train_features = char_feature.extract_all_features(train_domains)

    clf = RandomForestClassifier(n_estimators=800, criterion='entropy', max_features=18, random_state=0)
    clf.fit(train_features, train_labels)

    print("train")
    with open("/home/yandingkui/dga_detection/result_data/20180428/all_domain.json", "r") as F:
        map = json.loads(F.read())
    nx_domains = map.get("nx")
    l1_domains = []
    l1_real_domains = []
    l2_domains = []
    l2_real_domains = []
    for d in nx_domains:
        publicsuffix = MyPublicSuffixList.psl.publicsuffix(d)
        dp = d[:d.rindex(publicsuffix)                                                                                                                                                                           - 1]
        d_array = dp.split(".")
        if len(d_array) == 1:
            l1_domains.append(d_array[0])
            l1_real_domains.append(d)

        elif len(d_array) == 2:
            l2_domains.append(d_array)
            l2_real_domains.append(d)

    l1_features = char_feature.extract_all_features(l1_domains)
    l2_features = char_feature.extract_all_features_for_2(l2_domains)
    pred_r1 = clf.predict(l1_features)
    for i1 in range(len(pred_r1)):
        if pred_r1[i1] == 1:
            print(l1_real_domains[i1])

    pred_r2 = []
    for item in l2_features:
        rrr = clf.predict(item)
        if rrr[0] == 0 and rrr[1] == 0:
            pred_r2.append(0)
        else:
            pred_r2.append(1)
    for i2 in range(len(pred_r2)):
        if pred_r1[i2] == 1:
            print(l2_real_domains[i2])


if __name__ == "__main__":
    # nx
    # print("-----------NX------------")
    # print("my")
    # MY_expirement_process(m_file="split_AGDs", benign_file="split_benign_nx.json", n=820, m=26, c='entropy')
    # print("FANCI")
    #FANCI_expirement_process(m_file="split_AGDs", benign_file="split_benign_nx.json", n=800, m=18, c='entropy')

    # ac
    # print("-----------AC------------")
    # print("my")
    MY_expirement_process()
    #
    # print("FANCI")
    #FANCI_expirement_process()
    #
    # #wordlist
    # print("-----------WORD------------")
    # print("my")
    # MY_expirement_process(m_file="split_wordlist_AGDs.json", benign_file="split_benign_wordlist_ac.json", n=785, m=18, c='gini')
    # print("FANCI")
    #FANCI_expirement_process(m_file="split_wordlist_AGDs.json", benign_file="split_benign_wordlist_ac.json", n=760,m=18, c='entropy')
