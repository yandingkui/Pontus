import collections
import requests
import os
import pandas as pd
def filterData():
    count=0
    infectedIP=set()
    alldata=list()
    with open("../result_data/match_dga") as f:
        for line in f:
            lines=line.strip().split(",")
            alldata.append(lines)
            if lines[15]=="MULL":
                count=count+1
                infectedIP.add(lines[0])
    resultfile=open("../result_data/dga_A","a+")
    for lines in alldata:
        if (lines[0] in infectedIP) and (lines[15]!="MULL"):
            resultfile.write(",".join(lines)+"\n")
    resultfile.close()

def counter(filepath='../result_data/dga_A'):
    counter=collections.Counter()
    with open(filepath,"r") as f:
        for line in f:
            items=line.strip().split(",")
            counter[items[3]]=counter[items[3]]+1
    print(len(counter))
    print(counter)

def virus_total_test(d):
    url = 'https://www.virustotal.com/vtapi/v2/url/report'
    params = {'apikey': 'f76bdbc3755b5bafd4a18436bebf6a47d0aae6d2b4284f118077aa0dbdbd76a4',
              'resource': d}
    # params = {'apikey': '8380506dbd784fa6eec4ccdd8a1906979285f3c98df7265bff827a1920ba1a40',
    #           'resource': d}
    response = requests.get(url, params=params)
    flag = False
    scan = response.json().get("scans")
    info=""
    if scan != None:
        for (k, v) in scan.items():
            if v.get('detected') == True:
                flag = True
                info="{}:{}".format(k,v)
                break
    return flag,info,response.json()

def maliciousC2_test(filepath='../result_data/dga_A'):
    alldomains=set()
    with open(filepath,"r") as f:
        for line in f:
            items=line.strip().split(",")
            queryDomain=items[3].strip().lower()
            alldomains.add(queryDomain)
    hashandledomains=set()
    with open("./C2.log","r") as c2f:
        for r in c2f:
            rs=r.strip().split("#")
            hashandledomains.add(rs[0].strip().lower())
    handledomains=alldomains.difference(hashandledomains)
    print("totalnum:{}".format(len(handledomains)))
    logfile=open("./C2.log","a+")
    testnum=0
    for d in handledomains:
        flag,info,jsonmap=virus_total_test(d)
        logfile.write("{}#{}#{}#{}\n".format(d,flag,info,jsonmap))
        testnum=testnum+1
        print(testnum)
    logfile.close()



def getTotal():
    total = 0
    with open("./C2.log", "r") as f:
        for r in f:
            rs = r.strip().split("#")
            if rs[1] == "True":
                total = total + 1
    print(total)

def getType():
    root_dir = "/home/public/2019-01-07-dgarchive_full/"
    map=dict()
    for filename in os.listdir(root_dir):
        print(filename)
        type=filename[:filename.index(".csv")]
        df = pd.read_csv(os.path.join(root_dir, filename), header=None, error_bad_lines=False)
        for d in df.iloc[:, 0]:
            if map.__contains__(d):
                l=map.get(d)
                l.append(type)
            else:
                map[d]=[type]
    allType=set()
    with open("./C2.log", "r") as f:
        for r in f:
            rs = r.strip().split("#")
            if rs[1] == "True":
                for t in map.get(rs[0]):
                    allType.add(t)
    print(len(allType))
    print(allType)

#取得C2域名的A记录
def getAllArecords(result_file="../result_data/dga_virustotal_A"):

    resultFile=open(result_file,"w",encoding="utf-8")
    with open("../C2.log","r") as f:
        AGDs=[r.strip().split("#")[0].lower() for r in f]
    with open("../result_data/dga_A","r") as af:
        for line in af:
            lines=line.strip().split(",")
            if lines[3].strip() in AGDs:
                resultFile.write(line)
    resultFile.close()


if __name__=="__main__":
    getAllArecords()
