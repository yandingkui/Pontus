import sys
sys.path.append("..")
import json
from publicsuffixlist import PublicSuffixList
import numpy as np
import pandas as pd
from stringexperiment.char_feature import extract_all_features

def createdataset(type="train",AGD_file="../data_sets/split_AGDs",BD_file="../data_sets/split_benign_nx.json",datasetname="nx_train_data"):
    if type=="train":
        v_index=0
    else:
        v_index=1
    psl=PublicSuffixList()
    with open(AGD_file,"r") as f:
        AGD_dict=json.loads(f.read())
    with open(BD_file,"r") as f:
        bd_dict=json.loads(f.read())
    allAGDs=set()
    allBDs=set()
    for k,v in AGD_dict.items():
        for d in v[v_index]:
            pre_d=d[:d.rindex(psl.publicsuffix(d))-1]
            for l in pre_d.split("."):
                allAGDs.add(l)
    for d in bd_dict[type]:
        pre_d = d[:d.rindex(psl.publicsuffix(d)) - 1]
        for l in pre_d.split("."):
            allBDs.add(l)
    length=len(allAGDs)
    print(length)
    allBDs=list(allBDs)[:length]
    allAGDs=list(allAGDs)
    alldomains=allAGDs+allBDs
    alllabels=list(np.ones(length))+list(np.zeros(length))
    allfeatures=extract_all_features(alldomains)
    np.save("../data_sets/{}_features.npy".format(datasetname),allfeatures)
    data=dict()
    data["domains"]=pd.Series(alldomains,dtype='str')
    data["labels"]=pd.Series(alllabels,dtype='int32')
    df=pd.DataFrame(data=data)
    df.to_csv("../data_sets/{}.csv".format(datasetname),index=False)

def readTrainNXDs():
    features=np.load("../data_sets/nx_train_data_features.npy")
    print(features.shape)


if __name__=="__main__":
    createdataset(type="pred", AGD_file="../data_sets/split_AGDs",
                  BD_file="../data_sets/split_benign_nx.json",
                  datasetname="nx_pred_data")