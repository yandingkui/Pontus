

def filterData():
    count=0
    infectedIP=set()
    with open("../result_data/match_dga") as f:
        for line in f:
            lines=line.strip().split(",")
            if lines[15]=="MULL":
                count=count+1
                infectedIP.add(lines[0])
    print(count)
    print(len(infectedIP))
    for ip in infectedIP:
        print(ip)

if __name__=="__main__":
    filterData()


