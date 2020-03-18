import  random

def getDomains():

    with open("../dataprocess/C2.log","r") as f:
        DGAs=[ r.strip().split("#")[0] for r in f]

    with open("/home/yandingkui/Pontus_old/data_sets/Aleax","r") as fb:
        benignDomains=[r.strip() for r in fb]

    benignDomains=random.sample(benignDomains,len(DGAs))
    lastNum=int(len(DGAs)*0.8)
    random.shuffle(DGAs)
    random.shuffle(benignDomains)
    return DGAs[:lastNum], DGAs[lastNum:],benignDomains[:lastNum],benignDomains[lastNum:]



if __name__=="__main__":
    ds=["a","v","b","kkkk"]
    ds1=["c"]
    print(ds+ds1)


