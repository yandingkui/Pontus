import sys
sys.path.append("..")
import threading
from publicsuffixlist import PublicSuffixList

class PublicSuffix(object):
    _instance_lock = threading.Lock()
    _psl=None
    def __init__(self):
        _psl=PublicSuffixList(source="../source/public_suffix_list.dat",accept_unknown=False)

    def getPublicSuffixList(self,number):
        return self._psl

    def __new__(cls, *args, **kwargs):
        if not hasattr(PublicSuffix, "_instance"):
            with PublicSuffix._instance_lock:
                if not hasattr(PublicSuffix, "_instance"):
                    PublicSuffix._instance = object.__new__(cls)
        return PublicSuffix._instance
