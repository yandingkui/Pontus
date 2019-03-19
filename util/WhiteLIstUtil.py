import sys
sys.path.append("..")
import threading
import pandas as pd

class WhiteListUtil():
    _instance_lock = threading.Lock()

    def __init__(self,topN):
        """
        :param topN: int, adopt topN domain names from Aleax Top list

        """
        self._suffixset=set()
        atdomains=pd.read_csv("../source/top-1m.csv",nrows=topN,header=None,error_bad_lines=False)
        self._suffixset.update(atdomains.iloc[:,1])
        cdndomains=pd.read_csv("../source/top-1m.csv",header=None,error_bad_lines=False)
        self._suffixset.update(cdndomains.iloc[:, 1])
        commondomains=['in-addr.arpa','ip6.arpa', 'mcafee.com','redhat.com','root-servers.net']
        self._suffixset.update(commondomains)



    def getBenignSuffix(self):
        return self._suffixset

    def __new__(cls, *args, **kwargs):
        if not hasattr(WhiteListUtil, "_instance"):
            with WhiteListUtil._instance_lock:
                if not hasattr(WhiteListUtil, "_instance"):
                    WhiteListUtil._instance = object.__new__(cls)
        return WhiteListUtil._instance