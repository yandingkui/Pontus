import sys
sys.path.append("..")
from util.PublicSuffix import PublicSuffix

class SingleFilter():
    """

    """

    def haveCorrectTLD(self, domain: str):
        """
        judge a domain name whether has a valid TLD
        :param domain:str, a domain name
        :return: bool
        """
        psl=PublicSuffix().getPublicSuffixList()
        if psl.publicsuffix(domain)==None:
            return False
        else:
            return True

    def AleaxTopN(self,number):
        pass

class Filter():
    pass

if __name__=="__main__":
    sf=SingleFilter()
    test_domain=["domain.ddd",]
    for d in test_domain:
        print(sf.haveCorrectTLD(d))