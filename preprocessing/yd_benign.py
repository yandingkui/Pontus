import sys,os
sys.path.append("..")
from preprocessing.DomainFilter import Filter
from publicsuffixlist import PublicSuffixList
import numpy as np
import random
import json

def getBenign(filepath):
    psl=PublicSuffixList()
    filter=Filter()
    domains=[]

    # out=dict()
    # with open(filepath,"r") as f:
    #     for r in f:
    #         r_split=r.strip().split(":")
    #         if filter.inWhiteList(r_split[0]):
    #             pri=psl.privatesuffix(r_split[0])
    #             lll=out.get(pri)
    #             if lll is None:
    #                 lll=[]
    #             lll.append(r_split[0])
    #             out[pri]=lll
    #             continue
    #         domains.append(r_split[0])
    #
    #
    # num=0
    # break_flag=False
    # for i in range(9):
    #     for k,v in out.items():
    #         if i>=len(v) or k in ["aliyunduncc.com","360wzb.cn","yundunwaf.com","bugtags.com","wscloudcdn.com","ourdvsss.com","aliyundunwaf.com","aligfwaf.com"]:
    #             continue
    #         domains.append(v[i])
    #         num+=1
    #         if num>=311:
    #             break_flag=True
    #             break
    #     if break_flag:
    #         break

    with open(filepath,"r") as f:
        for r in f:
            r_split=r.strip().split(":")
            domains.append(r_split[0])
    random.shuffle(domains)

    result=dict()
    result["train"]=domains[:23600]
    result["pred"]=domains[23600:29500]

    with open("../result_data/yd_nf_data.json","w") as f:
        f.write(json.dumps(result))

    print(len(domains))


if __name__=="__main__":
    getBenign("/home/yandingkui/Pontus/result_data/yd_top")

