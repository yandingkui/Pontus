import collections

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
    print(counter.most_common(10))




if __name__=="__main__":
    counter()


