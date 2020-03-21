import  random

def getDomains():
    AGDs=[]
    with open("../C2.log", "r") as f:
        for r in f:
            items = r.strip().split("#")
            if items[1].strip() == "True":
                AGDs.append(items[0].strip())

    with open("/home/yandingkui/Pontus_old/data_sets/Aleax","r") as fb:
        benignDomains=[r.strip() for r in fb]
    random.shuffle(AGDs)
    for d in AGDs:
        print(d)
    benignDomains=random.sample(benignDomains,len(AGDs))
    lastNum=int(len(AGDs)*0.8)
    random.shuffle(AGDs)
    random.shuffle(benignDomains)
    return AGDs[:lastNum], AGDs[lastNum:],benignDomains[:lastNum],benignDomains[lastNum:]



if __name__=="__main__":
    with open("../dataprocess/C2.log","r") as f:
        DGAs=[ r.strip().split("#")[0] for r in f]
    print(DGAs)