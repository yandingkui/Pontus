import sys, os

sys.path.append("..")
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time
import numpy as np

np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from stringexperiment.pontus import pontus
from stringexperiment.char_feature import extract_all_features
from stringexperiment.pontus import pontus
from sklearn.metrics import accuracy_score
import json
import random


class generalization():
    def __init__(self):
        self.pontus = pontus()

    def getAllDomains(self, benignFile, AGDfile="../data_sets/AGD.json", ratio=1):
        with open(AGDfile, "r") as f:
            map = json.loads(f.read())
        DGADomains = []
        for k, v in map.items():
            DGADomains = DGADomains + v
        with open(benignFile, "r") as f:
            benign_domains = [r.strip() for r in f]
        return DGADomains, benign_domains

    def sameISP(self, ISP="yd"):
        clf = GradientBoostingClassifier(max_depth=9, n_estimators=150, max_features=32)
        root_dir = "../data_sets/"
        bfiles = []
        result=[]
        for filename in os.listdir(root_dir):
            if ISP in filename:
                bfiles.append(filename)
        for i in range(7):
            Tdga, a, Tben, b = self.pontus.getTrainTestDomains(benignFile="{}{}".format(root_dir, bfiles[i]),AGDfile="{}AGD{}.json".format(root_dir, i))
            Tdga_set=set(Tdga)
            Tben_set=set(Tben)
            X_train,y_train=self.pontus.get_dataset(Tdga,Tben)
            clf.fit(X_train,y_train)
            for j in range(7):
                if i == j:
                    continue
                else:
                    D, B = self.getAllDomains(benignFile="{}{}".format(root_dir, bfiles[j]),AGDfile="{}AGD{}.json".format(root_dir, j))
                    pdga_set=set(D)
                    pben_set=set(B)
                    pdga=list(pdga_set.difference(Tdga_set))
                    pben=list(pben_set.difference(Tben_set))

                    if len(pdga)>=len(pben):
                        number=len(pben)
                        pdga=random.sample(pdga,number)

                    else:
                        number=len(pdga)
                        pben=random.sample(pben,number)

                    X_test, y_test = self.pontus.get_dataset(pdga, pben)
                    print(len(y_test))
                    y_test_pred=clf.predict(X_test)
                    result.append(accuracy_score(y_test,y_test_pred))
        resultArray=np.asarray(result)
        print(resultArray)
        print("max={:.3f},min={:.3f},mean={:.3f},std={:.3f}".format(resultArray.max(),resultArray.min(),resultArray.mean(),resultArray.std()))

    def differentISP(self, ISP="dx",testISP="yd"):
        clf = GradientBoostingClassifier(max_depth=9, n_estimators=150, max_features=32)
        root_dir = "../data_sets/"
        model_bfiles = []
        test_bfiles = []
        for filename in os.listdir(root_dir):
            if ISP in filename:
                model_bfiles.append(filename)
            if testISP in filename:
                test_bfiles.append(filename)
        result=[]
        for i in range(7):
            Tdga, a, Tben, b = self.pontus.getTrainTestDomains(benignFile="{}{}".format(root_dir, model_bfiles[i]),
                                                               AGDfile="{}AGD{}.json".format(root_dir, i))
            Tdga_set = set(Tdga)
            Tben_set = set(Tben)
            X_train, y_train = self.pontus.get_dataset(Tdga, Tben)
            clf.fit(X_train, y_train)
            for j in range(7):
                D, B = self.getAllDomains(benignFile="{}{}".format(root_dir, test_bfiles[j]),
                                          AGDfile="{}AGD{}.json".format(root_dir, j))
                pdga_set = set(D)
                pben_set = set(B)
                pdga = list(pdga_set.difference(Tdga_set))
                pben = list(pben_set.difference(Tben_set))

                if len(pdga) >= len(pben):
                    number = len(pben)
                    pdga = random.sample(pdga, number)

                else:
                    number = len(pdga)
                    pben = random.sample(pben, number)

                X_test, y_test = self.pontus.get_dataset(pdga, pben)
                # print(len(y_test))
                y_test_pred = clf.predict(X_test)
                result.append(accuracy_score(y_test, y_test_pred))
        resultArray = np.asarray(result)
        print("{} -> {}".format(ISP,testISP))
        print("max={:.3f},min={:.3f},mean={:.3f},std={:.3f}".format(resultArray.max(), resultArray.min(),
                                                                    resultArray.mean(), resultArray.std()))


if __name__ == "__main__":
    G = generalization()
    G.sameISP(ISP="dx")
    print("--------------------------------------------------------------------------------------")
    G.differentISP(ISP="dx",testISP="yd")
    # G.differentISP(ISP="yd", testISP="dx")
