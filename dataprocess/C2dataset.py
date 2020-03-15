

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



if __name__=="__main__":
    filterData()


