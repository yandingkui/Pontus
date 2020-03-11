import pandas as pd
import os

from multiprocessing import  Lock
class Match():
    def __init__(self):
        self.DGADomains=set()
        self.root_dir = "/home/public/2019-01-07-dgarchive_full/"
        for filename in os.listdir(self.root_dir):
            df = pd.read_csv(os.path.join(self.root_dir, filename), header=None, error_bad_lines=False)
            for d in df.iloc[:, 0]:
                self.DGADomains.add(d)
        self.lock=Lock()

    def judge(self,domain):
        if(domain in self.DGADomains):
            return True
        else:
            return False

    def saveFile(self,line,filepath="../result_data/match_dga"):
        self.lock.acquire()
        with open(filepath,"a+") as f:
            f.write(line+"\n")
        self.lock.release()
