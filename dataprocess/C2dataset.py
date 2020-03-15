import collections
import requests

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

    if scan != None:
        for (k, v) in scan.items():
            if v.get('detected') == True:
                flag = True
                information="{}:{}".format(k,v)
                break
    return flag,information,response.json()

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


if __name__=="__main__":
    maliciousC2_test()
