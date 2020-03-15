

def filterData():
    count=0
    with open("../result_data/match_dga") as f:
        for line in f:
            lines=line.strip().split(",")
            if lines[15]=="MULL":
                count=count+1
    print(count)

if __name__=="__main__":
    filterData()


