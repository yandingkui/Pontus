import pandas as pd
import os
class Match():
    def __init__(self):
        self.DGADomains=set()
        self.root_dir = "/home/public/2019-01-07-dgarchive_full/"
        for filename in os.listdir(self.root_dir):
            df = pd.read_csv(os.path.join(self.root_dir, filename), header=None, error_bad_lines=False)
            for d in df.iloc[:, 0]:
                self.DGADomains.add(d)
        print(len(self.DGADomains))






if __name__=="__main__":
    match=Match()


