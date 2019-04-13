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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stringexperiment import char_feature
from publicsuffixlist import PublicSuffixList
###############################



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
    psl=PublicSuffixList()
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
            d_split = d[:d.index(psl.publicsuffix(d)) - 1].split(".")
            if len(d_split) == 1:
                train_domains.append(d_split[0])
            else:
                m = 0
                lm = None
                for l in d_split:
                    if len(l) > m:
                        lm = l
                train_domains.append(lm)
            train_labels.append(1)
        for d in v[1]:
            pred_domains.append(d)
            pred_labels.append(1)

    for d in benign_data.get("train"):
        pri_d=psl.privatesuffix(d)
        lm=pri_d[:pri_d.index(psl.publicsuffix(pri_d))-1]
        train_domains.append(lm)
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