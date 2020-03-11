import sys, os

sys.path.append("..")
from publicsuffixlist import PublicSuffixList
import json
import pandas as pd
import random
from util.MyJsonEncoder import MyJsonEncoder
from collections import Counter

def static_num(file_path):
    psl = PublicSuffixList()
    result = [0, 0, 0]
    with open(file_path, "r") as f:
        for r in f:
            d = r.strip().split(",")[0]
            d_strip = d[:d.rindex(psl.publicsuffix(d)) - 1].split(".")
            if len(d_strip) == 1:
                result[0] += 1
            elif len(d_strip) == 2:
                result[1] += 1
            else:
                result[2] += 1
    print(result)


def dga_static_num(file_path):
    psl = PublicSuffixList()
    result = [0, 0, 0]
    with open(file_path, "r") as f:
        map = json.loads(f.read())
    for k, v in map.items():
        for d in v[0]:

            d_strip = d[:d.rindex(psl.publicsuffix(d)) - 1].split(".")
            if len(d_strip) == 1:
                result[0] += 1
            elif len(d_strip) == 2:
                result[1] += 1
            else:
                result[2] += 1
        for d in v[1]:
            d_strip = d[:d.rindex(psl.publicsuffix(d)) - 1].split(".")
            if len(d_strip) == 1:
                result[0] += 1
            elif len(d_strip) == 2:
                result[1] += 1
            else:
                result[2] += 1
    print(result)


def get2subdomain(root_dir="/home/public/2019-01-07-dgarchive_full"):
    result = dict()
    psl = PublicSuffixList()
    for filename in os.listdir(root_dir):
        with open("{}/{}".format(root_dir, filename), "r") as f:
            for r in f:
                d = r.strip().split(",")[0]
                d_strip = d[:d.rindex(psl.publicsuffix(d)) - 1].split(".")
                if len(d_strip) == 2:
                    domains = result.get(filename)
                    if domains is None:
                        domains = set()
                        result[filename] = domains
                    domains.add(d)

    for k, v in result.items():
        print("{} : {}".format(k, len(v)))
        v_list = list(v)
        print(v_list[:10])



def static_1_2(root_dir="/home/public/2019-01-07-dgarchive_full"):
    psl=PublicSuffixList()
    result=dict()
    for filename in os.listdir(root_dir):
        df = pd.read_csv(os.path.join(root_dir,filename),header=None,error_bad_lines=False)
        domains = result.get(filename)
        if domains is None:
            domains = [set(), set()]
            result[filename] = domains
        for d in df.iloc[:,0]:
            pub_d=psl.publicsuffix(d)
            if d != pub_d:
                d_split=d[:d.rindex(pub_d)-1].split(".")
                if len(d_split)==1:
                    result.get(filename)[0].add(d)
                elif len(d_split)==2:
                    result.get(filename)[1].add(d)
                else:
                    print("Wow : {}".format(d))
        print("{} finish".format(filename))

    print("write")
    with open("../result_data/dga_data.json","w") as f:
        f.write(json.dumps(result,cls=MyJsonEncoder))


def analysis_dga():
    with open("../result_data/dga_data.json","r") as f:
        map=json.loads(f.read())
    number=0
    for k,v in map.items():
        if len(v[1])!=0:
            number+=1
            print("{}: {}".format(k,len(v[0])+len(v[1])))
            print(v[1][:10])
        else:
            if len(v[0])+len(v[1])>=1000:
                number+=1
    print(number)


def get_dga_data():
    result=dict()
    with open("../result_data/dga_data.json","r") as f:
        map=json.loads(f.read())
    for k,v in map.items():
        vl0=len(v[0])
        vl1=len(v[1])
        if vl1 != 0:
            if vl0==0:
                if vl1>500:
                    number=500
                else:
                    number=vl1
                result[k]= random.sample(v[1],number)
            else:
                if vl1>400:
                    number=400
                else:
                    number=vl1
                v1= random.sample(v[1],number)
                if vl0>500-number:
                    number=500-number
                else:
                    number=vl0
                v0= random.sample(v[0],number)
                result[k]=v0+v1
        else:
            if vl0>=1000:
                result[k]=random.sample(v[0],500)
    with open("../result_data/AGD.json","w") as f:
        f.write(json.dumps(result))



def get_counter(days=["20180427"]):
    root_dir="/home/yandingkui/Pontus/result_data/temp/"
    for day in days:
        counter=Counter()
        for i in range(24):
            path=os.path.join(root_dir,"{}{}.json".format(day,i))
            if os.path.exists(path):
                with open(path, "r") as f:
                    counter1 = Counter(json.loads(f.read()))
                counter.update(counter1)
        print("write")

        with open("{}{}.json".format(root_dir,day),"w") as f:
            f.write(json.dumps(counter))
        print("finish")

def get_most_common(self):

    dx_rootDir="../result_data/gddx/"
    yd_rootDir="../result_data/temp/"
    dx_days=["20171031","20171102","20171104","20171106","20171101","20171103","20171105"]
    yd_days=["20180427","20180428", "20180429","20180430","20180501"]

    for day in dx_days:
        domains=[]
        with open("{}{}{}".format(dx_rootDir,day,".json"),"r") as f:
            counter=Counter(json.loads(f.read()))
            for item in counter.most_common(30000):
                domains.append(item[0])
        with open("../data_sets/dx_{}".format(day),"w") as f:
            f.write("\n".join(domains))
    for day in yd_days:
        domains=[]
        with open("{}{}{}".format(yd_rootDir,day,".json"),"r") as f:
            counter=Counter(json.loads(f.read()))
            for item in counter.most_common(30000):
                domains.append(item[0])
        with open("../data_sets/yd_{}".format(day), "w") as f:
            f.write("\n".join(domains))



def DGA_dataSets():
    result=dict()
    with open("../result_data/dga_data.json","r") as f:
        map=json.loads(f.read())
    for i in range(7):
        for k,v in map.items():
            vl0=len(v[0])
            vl1=len(v[1])
            if vl1 != 0:
                if vl0==0:
                    if vl1>500:
                        number=500
                    else:
                        number=vl1
                    result[k]= random.sample(v[1],number)
                else:
                    if vl1>400:
                        number=400
                    else:
                        number=vl1
                    v1= random.sample(v[1],number)
                    if vl0>500-number:
                        number=500-number
                    else:
                        number=vl0
                    v0= random.sample(v[0],number)
                    result[k]=v0+v1
            else:
                if vl0>=1000:
                    result[k]=random.sample(v[0],500)
        with open("../data_sets/AGD{}.json".format(i),"w") as f:
            f.write(json.dumps(result))


def TwoLD_DataSet():
    with open("../result_data/dga_data.json", "r") as f:
        map = json.loads(f.read())
    all2LDAGD=[]
    for k,v in map.items():
        all2LDAGD=all2LDAGD+v[0]
    df=pd.read_csv("../source/top-1m.csv",error_bad_lines=False,header=None)
    domains=[]
    for d in df.iloc[:,1]:
        domains.append(d)
    print(len(domains))
    allBenign=domains[100001:]

    with open("../data_sets/Aleax","w") as f:
        f.write("\n".join(allBenign))
    with open("../data_sets/all2LDAGD","w") as f:
        f.write("\n".join(all2LDAGD))

def filter2LDAleax():
    psl=PublicSuffixList()
    data=[]
    with open("../data_sets/Aleax","r") as f:
        for r in f:
            d=r.strip()

            d1=d[:d.rindex(psl.publicsuffix(d))-1]
            if len(d1)==0:
                continue
            d_split=d1.split(".")
            if len(d_split)==1 and len(d_split[0])!=0:
                data.append(d)
        print(len(data))
    with open("../data_sets/Aleax2LD","w") as f:
        f.write("\n".join(data))


def get_word_list():
    with open("../result_data/dga_data.json", "r") as f:
        map = json.loads(f.read())
    result=dict()
    for k, v in map.items():
        if k=="gozi_dga.csv":
            result["gozi"]=random.sample(v[0]+v[1],10000)
        elif k=="matsnu_dga.csv":
            result["matsnu"]=random.sample(v[0]+v[1],10000)
        elif k=="suppobox_dga.csv":
            result["suppobox"]=random.sample(v[0]+v[1],10000)
        else:
            continue
    with open("../data_sets/wordlist.json","w") as f:
        f.write(json.dumps(result))



if __name__ == "__main__":
    # trainDGADomain,testDGADomain,trainBenignDomain,testBenignDomain=getSingleDataSet()
    # print(len(trainDGADomain))
    # print(len(testDGADomain))
    # print(len(trainBenignDomain))
    # print(len(testBenignDomain))
    # filter2LDAleax()

    get_word_list()
