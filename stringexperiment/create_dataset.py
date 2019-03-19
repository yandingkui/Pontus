import sys
sys.path.append("..")
import json
from publicsuffixlist import PublicSuffixList
import numpy as np
import pandas as pd
from stringexperiment.char_feature import extract_all_features

def createNXdataset():
    psl=PublicSuffixList()
    with open("../data_sets/split_AGDs","r") as f:
        AGD_dict=json.loads(f.read())
    with open("../data_sets/split_benign_nx.json","r") as f:
        bnx_dict=json.loads(f.read())
    allAGDs=set()
    allBenignNXDs=set()
    for k,v in AGD_dict.items():
        print(len(v[0]))
        for d in v[0]:
            pre_d=d[:d.rindex(psl.publicsuffix(d))-1]
            for l in pre_d.split("."):
                allAGDs.add(l)
    for d in bnx_dict["train"]:
        pre_d = d[:d.rindex(psl.publicsuffix(d)) - 1]
        for l in pre_d.split("."):
            allBenignNXDs.add(l)
    length=len(allAGDs)
    allBenignNXDs=list(allBenignNXDs)[:length]
    allAGDs=list(allAGDs)
    alldomains=allAGDs+allBenignNXDs
    alllabels=list(np.ones(length))+list(np.zeros(length))
    allfeatures=extract_all_features(alldomains)

    data=dict()
    data["domains"]=alldomains
    data["features"]=list(allfeatures)
    data["labels"]=alllabels

    df=pd.DataFrame(data=data)
    df.to_csv("../data_sets/nx_train_data.csv")


if __name__=="__main__":
    createNXdataset()