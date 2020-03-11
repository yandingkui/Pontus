import sys

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


class classifierSelection():
    def __init__(self):
        self.pontus = pontus()

    def get_dataset(self, bfile):
        trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = self.pontus.getTrainTestDomains(
            benignFile="../data_sets/{}".format(bfile))
        trainData = trainDGADomain + trainBenignDomain

        y_train = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))

        testData = testDGADomain + testBenignDomain
        y_test = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

        allLabels, index = self.pontus.getAllDomainLabels(trainData)
        labelFeatures = extract_all_features(allLabels)
        X_train = self.pontus.unionFeature(labelFeatures, index)

        allLabels, index = self.pontus.getAllDomainLabels(testData)
        labelFeatures = extract_all_features(allLabels)
        X_test = self.pontus.unionFeature(labelFeatures, index)

        return X_train, y_train, X_test, y_test

    def get_FPR_TPR(self, clf, X_train, y_train, X_test, y_test, c):
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        print("{} {}".format(c, end - start))
        pred_pr = clf.predict_proba(X_test)
        # y_test_pred=clf.predict(X_test)
        # print("{} performance".format(c))
        # self.pontus.printMetric(y_test,y_test_pred)

        pred_p = pred_pr[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pred_p)
        return fpr, tpr

    def draw_classifier_ROC(self, bfile):
        X_train, y_train, X_test, y_test = self.get_dataset(bfile)
        category = ["GBDT", "SVM", "RF"]

        plt.figure()
        plt.xlim(0, 0.2)
        plt.ylim(0.9, 1)
        for c in category:
            if c == "GBDT":
                clf = GradientBoostingClassifier(max_depth=9, n_estimators=150, max_features=32)
                fpr, tpr = self.get_FPR_TPR(clf, X_train, y_train, X_test, y_test, c)
                plt.plot(fpr, tpr, label=c, color='red', linestyle='-')

            elif c == "RF":
                clf = RandomForestClassifier(criterion='gini', max_features=24, n_estimators=840, random_state=0)
                fpr, tpr = self.get_FPR_TPR(clf, X_train, y_train, X_test, y_test, c)
                plt.plot(fpr, tpr, label=c, color='green', linestyle='--')
            else:
                clf = SVC(C=18, kernel='poly', probability=True)
                fpr, tpr = self.get_FPR_TPR(clf, X_train, y_train, X_test, y_test, c)
                plt.plot(fpr, tpr, label=c, color='purple', linestyle=':')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc="lower right")
        plt.savefig("../result_data/cs_{}.eps".format(bfile), format='eps', dpi=1000, bbox_inches='tight')

    def classifierSelection(self):
        bfiles = [ "dx_20171105"]
        for bfile in bfiles:
            self.draw_classifier_ROC(bfile)


if __name__ == "__main__":
    roc = classifierSelection()
    roc.classifierSelection()
