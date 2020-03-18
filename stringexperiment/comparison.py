# -*- coding: UTF-8 -*-
import sys
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
from FANCI import FANCI_features
from sklearn.externals import joblib
import numpy as np
from stringexperiment.classifierSelection import classifierSelection
from stringexperiment.pontus import pontus
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter
from activedomain import DataSetDomains

class comparison():

    def __init__(self):
        self.pontus = pontus()

    def FANCI_expirement_process(self, n=755, m=28, c='gini'):
        p = pontus()
        root_dir = "../data_sets/"
        yd_bfiles = []
        dx_bfiles = []
        for filename in os.listdir(root_dir):
            if "yd" in filename:
                yd_bfiles.append(filename)
            if "dx" in filename:
                dx_bfiles.append(filename)

        for i in range(1):
            bfiles = "{}{}".format(root_dir, dx_bfiles[i])
            AGDfile = "{}AGD{}.json".format(root_dir, i)
            trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = p.getTrainTestDomains(
                benignFile=bfiles,
                AGDfile=AGDfile)
            trainData = trainDGADomain + trainBenignDomain
            trainLabel = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))
            testData = testDGADomain + testBenignDomain
            testLabel = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

            train_features = FANCI_features.extract_all_features(trainData)

            clf = RandomForestClassifier(n_estimators=n, max_features=m, criterion=c)
            clf.fit(train_features, trainLabel)

            pred_features = FANCI_features.extract_all_features(testData)

            predict_result = clf.predict(pred_features)

            print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}" \
                  .format(accuracy_score(testLabel, predict_result), \
                          recall_score(testLabel, predict_result), \
                          precision_score(testLabel, predict_result), \
                          f1_score(testLabel, predict_result)))

    def perdiction(self, trainDomains, trainLabel, preictDomains, preictLabels, type="FANCI"):
        if type == "FANCI":
            train_features = FANCI_features.extract_all_features(trainDomains)
            clf = RandomForestClassifier(n_estimators=755, max_features=28, criterion='gini')
            clf.fit(train_features, trainLabel)
            pred_features = FANCI_features.extract_all_features(preictDomains)

        else:
            train_features = self.pontus.getDomainFeatures(trainDomains)
            clf = GradientBoostingClassifier(max_depth=9, n_estimators=150, max_features=32)
            clf.fit(train_features, trainLabel)
            pred_features = self.pontus.getDomainFeatures(preictDomains)

        predict_result = clf.predict(pred_features)

        # if type == "pontus":
        #     for i in range(len(preictDomains)):
        #         if predict_result[i] == 1 and preictLabels[i] == 0:
        #             print(preictDomains[i])
        #
        # print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}" \
        #       .format(accuracy_score(preictLabels, predict_result), \
        #               recall_score(preictLabels, predict_result), \
        #               precision_score(preictLabels, predict_result), \
        #               f1_score(preictLabels, predict_result)))

        self.pontus.printMetric(preictLabels,predict_result)

    def testAleax2LD(self):
        p = pontus()
        root_dir = "../data_sets/"
        bfiles = "{}{}".format(root_dir, "yd_20180430")
        AGDfile = "{}AGD{}.json".format(root_dir, 5)
        trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = p.getTrainTestDomains(benignFile=bfiles,
                                                                                                   AGDfile=AGDfile,
                                                                                                   ratio=1)
        sd = set(trainDGADomain)
        sb = set(trainBenignDomain)

        mdga = set()
        with open(os.path.join(root_dir, "all2LDAGD"), "r") as f:
            for r in f:
                mdga.add(r.strip())

        b2ld = set()
        with open(os.path.join(root_dir, "Aleax2LD"), "r") as f:
            for r in f:
                b2ld.add(r.strip())

        testDGADomain = list(mdga.difference(sd))[:20000]

        testBenignDomain = list(b2ld.difference(sb))[:20000]

        trainDomains = trainDGADomain + trainBenignDomain
        trainLabel = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))

        testDomains = testDGADomain + testBenignDomain
        testLabel = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

        # for i in range(len(testDomains)):
        #     print("{} {}".format(testDomains[i],testLabel[i]))
        # print("FANCI")

        # self.perdiction(trainDomains,trainLabel,testDomains,testLabel,"FANCI")
        self.perdiction(trainDomains, trainLabel, testDomains, testLabel, "pontus")

    def classification_comparasion(self,bfile="yd_20180430", AGDfile="AGD1.json"):
        root_dir = "../data_sets/"
        bfile = "{}{}".format(root_dir, bfile)
        AGDfile = "{}{}".format(root_dir, AGDfile)
        trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = self.pontus.getTrainTestDomains(
            benignFile=bfile,
            AGDfile=AGDfile,
            ratio=0.8)
        trainDomains = trainDGADomain + trainBenignDomain
        trainLabel = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))

        preictDomains = testDGADomain + testBenignDomain
        preictLabels = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

        self.perdiction(trainDomains, trainLabel, preictDomains, preictLabels)
        self.perdiction(trainDomains, trainLabel, preictDomains, preictLabels,type="pontus")

    def addConfusion(self):
        root_dir = "../data_sets/"
        # bfiles = "{}{}".format(root_dir, "yd_20180430")
        # AGDfile = "{}AGD{}.json".format(root_dir, 0)
        # trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = self.pontus.getTrainTestDomains(
        #     benignFile=bfiles,
        #     AGDfile=AGDfile,
        #     ratio=0.8)
        # sd = set(trainDGADomain)
        # sb = set(trainBenignDomain)

        mdga = []
        with open(os.path.join(root_dir, "all2LDAGD"), "r") as f:
            for r in f:
                mdga.append(r.strip())

        b2ld = []
        with open(os.path.join(root_dir, "Aleax2LD"), "r") as f:
            for r in f:
                b2ld.append(r.strip())

        random.shuffle(mdga)
        random.shuffle(b2ld)



        trainDGADomain=mdga[:6000]
        trainBenignDomain=b2ld[:6000]
        trainDomains = trainDGADomain + trainBenignDomain
        trainLabel = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))
        testDGADomain=mdga[6000:20000]
        testBenignDomain=b2ld[6000:20000]
        testDomains=testDGADomain+testBenignDomain
        testLabel = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

        train_features = self.pontus.getDomainFeatures(trainDomains)
        FANCI_train_features=FANCI_features.extract_all_features(trainDomains)
        clf = GradientBoostingClassifier(max_depth=9, n_estimators=150, max_features=32)
        FANCI = RandomForestClassifier(n_estimators=55, max_features=8, criterion='gini')
        clf.fit(train_features, trainLabel)
        FANCI.fit(FANCI_train_features, trainLabel)

        pred_features = self.pontus.getDomainFeatures(testDomains)
        FANCI_pred_features=FANCI_features.extract_all_features(testDomains)

        pred_result = clf.predict(pred_features)
        FANCI_pred_result = FANCI.predict(FANCI_pred_features)
        print("Pontus:")
        self.pontus.printMetric(testLabel,pred_result)

        print("FANCI:")
        self.pontus.printMetric(testLabel, FANCI_pred_result)

        # confusionTestDGAFeatures = self.pontus.getDomainFeatures(confusionTestDGADomain)
        # confusionTestBenignFeatures = self.pontus.getDomainFeatures(confusionTestBenignDomain)
        #
        # FANCI_confusionTestDGAFeatures=FANCI_features.extract_all_features(confusionTestDGADomain)
        # FANCI_confusionTestBenignFeatures=FANCI_features.extract_all_features(confusionTestBenignDomain)
        #
        # testLabel=list(testLabel)
        # last=0
        # for index in range(0, 20001, 1000):
        #     if index != 0:
        #         for i in range(last,index):
        #             pred_features.append(confusionTestDGAFeatures[i])
        #             testLabel.append(1)
        #             pred_features.append(confusionTestBenignFeatures[i])
        #             testLabel.append(0)
        #         FANCI_pred_features=np.append(FANCI_pred_features,FANCI_confusionTestDGAFeatures[last:index,:],axis=0)
        #         FANCI_pred_features = np.append(FANCI_pred_features, FANCI_confusionTestBenignFeatures[last:index, :],axis=0)
        #         last = index


            # print(len(testLabel))
            # print(len(pred_result))
            # print("{}  {}".format(accuracy_score(testLabel, pred_result),accuracy_score(testLabel,FANCI_pred_result)))


    def drawZhexian(self):

        x=range(0,40001,2000)
        pontus=[96.5, 93.6, 91.6, 89.8, 88.4, 87.2, 86.4, 85.6, 85.0, 84.3, 83.7, 83.2, 82.9, 82.6, 82.3, 82.0, 81.7, 81.5, 81.3, 81.1, 80.9]

        fanci=[92.4, 86.2, 81.6, 78.0, 75.2, 72.8, 70.8, 69.1, 67.8, 66.6, 65.5, 64.5, 63.6, 63.0, 62.3, 61.7, 61.2, 60.7, 60.2, 59.7, 59.3]

        plt.title('Result Analysis')
        plt.plot(x, fanci, color='green', label='FANCI')
        plt.plot(x, pontus, color='red', label='Pontus')

        plt.legend()  # 显示图例
        plt.grid()
        plt.xlabel('sample number')
        plt.ylabel('accuracy')
        plt.savefig("/Users/yandingkui/Desktop/confusion.eps", format='eps', dpi=1000, bbox_inches='tight')
        plt.show()

    def get_FPR_TPR(self, clf, X_train, y_train, X_test, y_test):

        clf.fit(X_train, y_train)

        rrr=clf.predict(X_test)
        self.pontus.printMetric(y_test,rrr)

        pred_pr = clf.predict_proba(X_test)
        pred_p = pred_pr[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pred_p)
        return fpr, tpr

    def word_list_roc(self):

        with open("../data_sets/wordlist.json","r") as f:
            map=json.loads(f.read())
        train_domains=[]
        pred_domains=[]
        for k,v in map.items():
            number=int(len(v)*0.80)
            train_domains=train_domains+v[:number]
            pred_domains=pred_domains+v[number:]

        AL=[]
        with open("../data_sets/Aleax2LD","r") as f:
            for r in f:
                AL.append(r.strip())

        random.shuffle(AL)
        TL=len(train_domains)
        train_labels=np.concatenate((np.ones(TL),np.zeros(TL)))
        train_domains=train_domains+AL[:TL]
        PL =len(pred_domains)
        pred_labels=np.concatenate((np.ones(PL),np.zeros(PL)))

        pred_domains=pred_domains+AL[TL:TL+PL]

        train_features = self.pontus.getDomainFeatures(train_domains)
        pred_features=self.pontus.getDomainFeatures(pred_domains)
        FANCI_train_features = FANCI_features.extract_all_features(train_domains)
        FANCI_pred_features=FANCI_features.extract_all_features(pred_domains)
        clf = GradientBoostingClassifier(max_depth=10, n_estimators=120, max_features=32)
        FANCI = RandomForestClassifier(n_estimators=755, max_features=28, criterion='gini')


        # print("pontus")
        pontus_fpr, pontus_tpr=self.get_FPR_TPR(clf,train_features,train_labels,pred_features,pred_labels)
        #
        # print(pontus_fpr)
        # print(pontus_tpr)
        #
        # print("FANCI")
        fanci_fpr, fanci_tpr=self.get_FPR_TPR(FANCI,FANCI_train_features,train_labels,FANCI_pred_features,pred_labels)

        mpl.rcParams['font.size'] = 17
        plt.xlim(0, 1)
        plt.ylim(0,1)
        plt.plot(pontus_fpr, pontus_tpr, label="Pontus", color='red', linestyle='-')
        plt.plot(fanci_fpr, fanci_tpr, label="FANCI", color='purple', linestyle=':')
        plt.xlabel('False Positive Rate',fontsize=17)
        plt.ylabel('True Positive Rate',fontsize=17)
        plt.legend(loc="lower right",fontsize=17)
        plt.savefig("../result_data/cs_wordlist.eps", format='eps', dpi=1000, bbox_inches='tight')
        print("finish")

    def testactive(self):
        # p = pontus()

        trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain =DataSetDomains.getDomains()

        trainDomains = trainDGADomain + trainBenignDomain
        trainLabel = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))

        testDomains = testDGADomain + testBenignDomain
        testLabel = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

        # for i in range(len(testDomains)):
        #     print("{} {}".format(testDomains[i],testLabel[i]))
        # print("FANCI")

        # self.perdiction(trainDomains,trainLabel,testDomains,testLabel,"FANCI")
        self.perdiction(trainDomains, trainLabel, testDomains, testLabel, "pontus")

    # def to_percent(self,temp, position):
    #     return '%1.0f' % (100 * temp) + '%'
    #
    # def CDF(self):
    #     allDGA = []
    #     with open("../data_sets/all2LDAGD", "r") as f:
    #         for r in f:
    #             allDGA.append(r.strip())
    #     benign = []
    #     with open("../data_sets/Aleax2LD", "r") as f:
    #         for r in f:
    #             benign.append(r.strip())
    #
    #     D = random.sample(allDGA, 1000)
    #     B = random.sample(benign, 1000)
    #
    #     result0 = []
    #     result1 = []
    #     for d in D:
    #         result0.append()
    #         # result1.append(_top_100k_readability(d))
    #     # print(result0)
    #     # print(max(result0))
    #     # print(min(result0))
    #     x = np.arange(0, 0.15, 0.005)
    #     cDGA = Counter()
    #
    #     cBen = Counter()
    #     for i in result0:
    #         for t in x:
    #             if i <= t:
    #                 cDGA[t] += 1
    #             else:
    #                 cDGA[t] += 0
    #
    #     result3 = []
    #     result4 = []
    #     for d in B:
    #         result3.append(_readability(d))
    #         # result4.append(_top_100k_readability(d))
    #
    #     for i in result3:
    #         for t in x:
    #             if i <= t:
    #                 cBen[t] += 1
    #             else:
    #                 cBen[t] += 0
    #
    #     y1 = []
    #     y2 = []
    #     for i in x:
    #         y1.append(float(cDGA.get(i)) / 1000.0)
    #         y2.append(float(cBen.get(i)) / 1000.0)
    #
    #     # plt.xlim(xmax=0.15, xmin=0)
    #     # plt.ylim(ymax=0.15, ymin=0)
    #
    #     plt.xlabel('the value of the Probability Summation Based on Article')
    #     plt.ylabel('ratio')
    #     plt.plot(x, y1, c="red", label="AGDs", linestyle='-', marker='4')
    #     plt.plot(x, y2, c="green", label="Benign domain names", linestyle=":")
    #
    #     plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #     plt.legend(loc="lower right")
    #
    #     plt.grid()  # 生成网格
    #     plt.savefig("../result_data/cs_{}.eps".format("sandian"),
    #                 format='eps', dpi=1000, bbox_inches='tight')

if __name__ == "__main__":
    C = comparison()
    # C.testAleax2LD()
    # print("dx")
    # C.classification_comparasion(bfile="dx_20171105",AGDfile="AGD6.json")
    # print("yd")
    # C.classification_comparasion(bfile="yd_20180427", AGDfile="AGD.json")
    # C.addConfusion()
    C.testactive()