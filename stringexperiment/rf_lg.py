# coding:utf-8
import sys, os

sys.path.append("..")
from multiprocessing import Pool, Manager
import subprocess
import traceback
from Levenshtein import distance
import pandas as pd
from util.MyJsonEncoder import MyJsonEncoder
from sklearn.metrics import roc_curve
import random
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stringexperiment import char_feature
from publicsuffixlist import PublicSuffixList
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from FANCI import FANCI_features
from sklearn import tree
from sklearn.model_selection import cross_val_score
import json
import numpy as np
import time


from sklearn.externals import joblib


class doubleClassifier():
    def __init__(self):
        # self.psl = PublicSuffixList()
        pass

    def getDomains(self, m_file_path, benign_file_path):
        with open(m_file_path, "r") as f:
            malicious_data = json.loads(f.read())
        with open(benign_file_path, "r") as f:
            benign_data = json.loads(f.read())

        LC_train_domain = []
        LC_train_labels = []

        DC_train_domain = []
        DC_train_labels = []

        pred_domain = []
        pred_labels = []
        for k, v in malicious_data.items():
            length = int(len(v[0]) * 0.8)
            for i in range(len(v[0])):
                d = v[0][i]
                if i < length:
                    LC_train_domain.append(d)
                    LC_train_labels.append(1)
                else:
                    DC_train_domain.append(d)
                    DC_train_labels.append(1)

            for d in v[1]:
                pred_domain.append(d)
                pred_labels.append(1)

        bdlist = benign_data.get("train")
        length = int(len(bdlist) * 0.8)

        for i in range(len(bdlist)):
            d = bdlist[i]
            if i < length:
                LC_train_domain.append(d)
                LC_train_labels.append(0)
            else:
                DC_train_domain.append(d)
                DC_train_labels.append(0)

        for d in benign_data.get("pred"):
            pred_domain.append(d)
            pred_labels.append(0)
        return LC_train_domain, LC_train_labels, DC_train_domain, DC_train_labels, pred_domain, pred_labels

    def getDomainClassifierLabels(self, domain, labels):
        domainLabels = []
        for i in range(len(labels)):
            d = domain[i]
            d_split = d[:d.rindex(self.psl.publicsuffix(d)) - 1].split(".")
            if labels[i] == 1:
                if len(d_split) == 1:
                    lm = d_split[0]
                else:
                    m = 0
                    lm = None
                    for l in d_split:
                        if len(l) > m:
                            lm = l
                domainLabels.append(lm)
            else:
                domainLabels.append(d_split[len(d_split) - 1])

        return domainLabels

    def getAllDomainLabels(self, domains):
        labels = []
        index = []
        psl=PublicSuffixList()
        for i in range(len(domains)):
            d = domains[i].strip()
            pub=psl.publicsuffix(d)
            d_split = d[:d.rindex(pub) - 1].split(".")
            if len(d_split) >2:
                print("d:{} pub:{}".format(d,pub))
            for l in d_split:
                if len(l) == 0:
                    print("kong kong")
                labels.append(l)
                index.append(i)
        return labels, index

    def getDomainLabelFeatures(self, domainlabels):
        return char_feature.extract_all_features(domainlabels)

    def label_classifier(self, domainlabel_features, domainlabel_labels):
        clf = RandomForestClassifier(criterion='entropy', max_features=14, n_estimators=760, random_state=0)
        scores = cross_val_score(clf, domainlabel_features, domainlabel_labels, cv=5)
        print("label classifier : {}".format(max(scores)))
        clf.fit(domainlabel_features, domainlabel_labels)
        return clf

    def domain_classifier(self, domain_features, domain_lebels):
        # clf = LogisticRegression(random_state=0, solver='lbfgs', ).fit(domain_features, domain_lebels)
        # clf = SVC(kernel='linear', C=1)
        clf = tree.DecisionTreeClassifier()
        scores = cross_val_score(clf, domain_features, domain_lebels, cv=5)
        print("domain classifier : {}".format(max(scores)))
        clf.fit(domain_features, domain_lebels)
        return clf

    def labelClassifierPredProba(self, clf, labelFeatures):
        result = []
        pred_result = clf.predict_proba(labelFeatures)
        for pr in pred_result:
            result.append(pr[0])
        return result

    def labelClassifierPred(self, clf, labelFeatures):
        return clf.predict(labelFeatures)

    def predictLabelFeatures(self, predresult, index):
        result = []
        i = 0
        while (True):
            if i == len(index) - 1:
                result.append([0, predresult[i]])
                i += 1
            elif i > len(index) - 1:
                break
            else:
                if index[i] == index[i + 1]:
                    result.append([predresult[i], predresult[i + 1]])
                    i += 2
                else:
                    result.append([-1, predresult[i]])
                    i += 1
        return np.asarray(result)

    def train(self, LC_domain, LC_label, DC_domain, DC_label):
        print("开始label分类器训练：{}  {}".format(len(LC_domain), len(LC_label)))

        train_domainlabels = self.getDomainClassifierLabels(LC_domain, LC_label)
        print("得到训练label {}".format(len(train_domainlabels)))

        domainlabel_features = self.getDomainLabelFeatures(train_domainlabels)
        print("提取label特征")
        labelClassifier = self.label_classifier(domainlabel_features, LC_label)
        print("训练分类器")

        print("开始域名分类器的训练：{}  {}".format(len(DC_domain), len(DC_label)))
        labels, index = self.getAllDomainLabels(DC_domain)
        print("提取label {}".format(len(labels)))

        trainAllDomainLabelFeatures = self.getDomainLabelFeatures(labels)

        labelProba = self.labelClassifierPredProba(labelClassifier, trainAllDomainLabelFeatures)
        print("label分类器进行打分")
        domainFeatures = self.predictLabelFeatures(labelProba, index)
        print("获得域名分类特征：{}  {}".format(len(DC_domain), len(domainFeatures)))
        domainClassifier = self.domain_classifier(domainFeatures, DC_label)
        print("训练域名分类器")

        return labelClassifier, domainClassifier
        # return 1,1

    def predict(self, labelFeatures, index, labelClassifier, domainClassifier):
        allLabelPorba = self.labelClassifierPredProba(labelClassifier, labelFeatures)
        domainFeatures = self.predictLabelFeatures(allLabelPorba, index)
        pred_result = domainClassifier.predict(domainFeatures)
        return pred_result

    def predictWithOutDomainClassifier(self, labelFeatures, index, labelClassifier):
        one_pred_result = labelClassifier.predict(labelFeatures)
        print("预测完毕")
        result = []
        i = 0
        while (True):
            if i == len(index) - 1:
                result.append(one_pred_result[i])
                i += 1
            elif i > len(index) - 1:
                break
            else:
                if index[i] == index[i + 1]:
                    if one_pred_result[i] == 1 or one_pred_result[i + 1] == 1:
                        result.append(1)
                    else:
                        result.append(0)
                    i += 2
                else:
                    result.append(one_pred_result[i])
                    i += 1

        print("统计完毕")
        return result

    def unionFeature(self, features, index):
        result = []
        i = 0
        while (True):
            if i == len(index) - 1:
                result.append(np.concatenate((np.zeros(len(features[i])), features[i])))
                i += 1
            elif i > len(index) - 1:
                break
            else:
                if index[i] == index[i + 1]:
                    result.append(np.concatenate((features[i], features[i + 1])))
                    i += 2
                else:
                    result.append(np.concatenate((np.zeros(len(features[i])), features[i])))
                    i += 1
        return result

    def printMetric(self, real, predict):
        print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}" \
              .format(accuracy_score(real, predict), \
                      recall_score(real, predict), \
                      precision_score(real, predict), \
                      f1_score(real, predict)))


    def unionClassify(self):
        starTime = time.time()
        # m_file = "/home/yandingkui/dga_detection/result_data/split_AGDs"
        # b_file = "/home/yandingkui/dga_detection/result_data/split_benign_ac.json"
        # b_file = "/home/yandingkui/Pontus/result_data/test_yd.json"
        # b_file="../result_data/yd_nf_data.json"

        trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = getSingleDataSet()
        trainData=trainDGADomain+trainBenignDomain

        trainLabel=np.concatenate((np.ones(len(trainDGADomain)),np.zeros(len(trainBenignDomain))))

        testData=testDGADomain+testBenignDomain
        testLabel=np.concatenate((np.ones(len(testDGADomain)),np.zeros(len(testBenignDomain))))

        # LC_domain, LC_label, DC_domain, DC_label, pred_domain, pred_labels = self.getDomains(m_file_path=m_file,
        #                                                                                  benign_file_path=b_file)


        allLabels, index = self.getAllDomainLabels(trainData)
        labelFeatures = self.getDomainLabelFeatures(allLabels)
        unionFeatures = self.unionFeature(labelFeatures, index)

        n_es_list = range(800, 900, 2)
        # max_fea_list = range(10, 30, 2)
        # max_depth_list=range(4,10)
        # tuned_parameters = [{'n_estimators': n_es_list, 'random_state': [0], 'max_features': max_fea_list,
        #                      'criterion': ["gini", "entropy"],'max_depth':max_depth_list}]

        # clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=38)
        # {'criterion': 'gini', 'max_depth': 9, 'max_features': 24, 'n_estimators': 840, 'random_state': 0}
        clf=RandomForestClassifier(criterion='gini', max_features=24, n_estimators=840, random_state=0)

        # clf = GradientBoostingClassifier(max_depth=9, n_estimators=150, max_features=32)

        # clf=joblib.load("../result_data/yd_test_train_model.m")

        # {'C': 1.5, 'kernel': 'linear'}
        # tuned_parameters = [{"C": [15,16,17,18],
        #                      "kernel": ['poly']}]
        # clf = GridSearchCV(SVC(), tuned_parameters, cv=3, scoring='accuracy', n_jobs=32)

        clf.fit(unionFeatures, trainLabel)
        # print(clf.best_params_)

        allLabels, index = self.getAllDomainLabels(testData)
        labelFeatures = self.getDomainLabelFeatures(allLabels)
        unionFeatures = self.unionFeature(labelFeatures, index)

        predict = clf.predict(unionFeatures)

        self.printMetric(testLabel, predict)

        joblib.dump(clf, "../result_data/yd_train_model.m")

    def classify(self):
        m_file = "/home/yandingkui/dga_detection/result_data/split_AGDs"
        b_file = "/home/yandingkui/dga_detection/result_data/split_benign_ac.json"
        LC_domain, LC_label, DC_domain, DC_label, pred_domain, pred_labels = self.getDomains(m_file_path=m_file,
                                                                                             benign_file_path=b_file)

        # labelclassifier, domainclassifier = self.train(LC_domain, LC_label, DC_domain, DC_label)
        #
        # allLabels, index = self.getAllDomainLabels(pred_domain)
        # labelFeatures = self.getDomainLabelFeatures(allLabels)
        # print("domain classifier:")
        # pred_result = self.predict(labelFeatures, index, labelclassifier, domainclassifier)
        # for i in range(len(pred_result)):
        #     if pred_result[i] != pred_labels[i]:
        #         print("{} pred:{} real:{} ".format(pred_domain[i],pred_result[i],pred_labels[i]))

        # print("0 1")
        # pred_result_one=self.predictWithOutDomainClassifier(labelFeatures, index, labelclassifier)


if __name__ == "__main__":
    dc = doubleClassifier()
    dc.unionClassify()
