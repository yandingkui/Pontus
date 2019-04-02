import sys,os
sys.path.append("..")
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from stringexperiment.char_feature import extract_all_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelExtractor():
    def save_model(self,training_data,training_labels,type,parameters):
        clf = RandomForestClassifier(n_estimators=parameters[0], max_features=parameters[1], criterion=parameters[2], random_state=0)
        clf.fit(training_data,training_labels)
        joblib.dump(clf, "../result_data/{}_model.m".format(type))

    def get_data(self,domain_file,feature_file):
        df=pd.read_csv(domain_file,error_bad_lines=False)
        features=np.load(feature_file)
        labels=np.array(df.loc[:,"labels"])
        return features,labels


    def obtain_model(self):
        types=["ac","nx"]
        paramters=[(815,10,'entropy'),(800,18,'entropy')]
        root_dir="../data_sets/"
        for i in range(2):
            t=types[i]
            p=paramters[i]
            features,labels=self.get_data(os.path.abspath("{}{}_train_data.csv".format(root_dir,t)),os.path.abspath("{}{}_train_data_features.npy".format(root_dir,t)))
            self.save_model(features,labels,type=t,parameters=p)


    def test_model(self,test_data,real_labels,type):
        print("{} model test result:".format(type))
        clf=joblib.load("../result_data/{}_model.m".format(type))
        pred_labels=clf.predict(test_data)
        print(pred_labels)
        print("accuracy:{}\nrecall:{}\nprecision:{}\nf1-score:{}" \
             .format(accuracy_score(pred_labels, real_labels), \
                     recall_score(pred_labels, real_labels), \
                     precision_score(pred_labels, real_labels), \
                     f1_score(pred_labels, real_labels)))


    def test(self):
        types=["ac","nx"]
        root_dir = "../data_sets/"
        for i in range(2):
            t=types[i]
            features, labels = self.get_data(os.path.abspath("{}{}_pred_data.csv".format(root_dir, t)),
                                             os.path.abspath("{}{}_pred_data_features.npy".format(root_dir, t)))
            self.test_model(features,labels,t)


if __name__=="__main__":
    modelextractor=ModelExtractor()
    # modelextractor.test()
    features=extract_all_features(["www","xxfeee0d8","validttu"])
    modelextractor.test_model(features,[0,1,0],"ac")