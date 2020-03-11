import sys

sys.path.append("..")
from publicsuffixlist import PublicSuffixList
import json
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stringexperiment import char_feature
from sklearn.externals import joblib

class pontus():




    def getTrainTestDomains(self,benignFile, AGDfile="../data_sets/AGD.json", ratio=0.8):
        with open(AGDfile, "r") as f:
            map = json.loads(f.read())
        trainDGADomain = []
        testDGADomain = []
        for k, v in map.items():
            number = int(len(v) * ratio)
            random.shuffle(v)
            trainDGADomain = trainDGADomain + v[:number]
            testDGADomain = testDGADomain + v[number:]

        train_number = len(trainDGADomain)
        test_number = len(testDGADomain)

        with open(benignFile, "r") as f:
            benign_domains = [r.strip() for r in f]
        random.shuffle(benign_domains)
        trainBenignDomain = benign_domains[:train_number]
        testBenignDomain = benign_domains[train_number:train_number + test_number]

        return trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain

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
                    print(d_split)
                    print("{} {} kong kong".format(d,pub))
                labels.append(l)
                index.append(i)
        return labels, index


    def get_dataset(self, DGADomain,benignDomain):
        domains = DGADomain + benignDomain
        y = np.concatenate((np.ones(len(DGADomain)), np.zeros(len(benignDomain))))
        allLabels, index = self.getAllDomainLabels(domains)
        labelFeatures = char_feature.extract_all_features(allLabels)
        X = self.unionFeature(labelFeatures, index)
        if len(X)!=len(y):
            print("error")
        return X,y

    def getDomainFeatures(self,domains):
        allLabels, index = self.getAllDomainLabels(domains)
        labelFeatures = char_feature.extract_all_features(allLabels)
        X = self.unionFeature(labelFeatures, index)
        return X

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

    def FPR(self,real, pred):
        fp = 0
        for index in range(len(real)):
            if real[index] == 0 and pred[index] == 1:
                fp += 1
        return 1-(fp * 1.0 / (len(real) / 2.0))

    def printMetric(self, real, predict):
        print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}\nFPR:{}" \
              .format(accuracy_score(real, predict), \
                      recall_score(real, predict), \
                      precision_score(real, predict), \
                      f1_score(real, predict),
                      self.FPR(real,predict)
                      ))
