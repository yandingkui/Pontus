class Match():
    def __init__(self):
        self.DGADomains=set()
        root_dir = "/home/public/2019-01-07-dgarchive_full/ud4_dga.csv"
        with open(root_dir,"r") as f:
            for r in f:
                domain=r.strip().split(",")[0].replace("\"")
                print(domain)




if __name__=="__main__":
    match=Match


