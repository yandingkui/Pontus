import sys
sys.path.append("..")
from util.PublicSuffix import PublicSuffix
from util.WhiteLIstUtil import WhiteListUtil
import re

class SingleFilter():

    def __init__(self,number):
        self.psl = PublicSuffix().getPublicSuffixList()
        self.wl=WhiteListUtil(number)

    def haveCorrectTLD(self, domain: str):
        """
        judge a domain name whether has a valid TLD
        :param domain:str, a domain name
        :return: bool
        """
        if self.psl.publicsuffix(domain)==None:
            return False
        else:
            return True

    def inWhiteList(self,domain:str):
        """
        :param domain: str
        :return:
        """
        benignSuffix=self.wl.getBenignSuffix()
        if self.psl.privatesuffix(domain) in benignSuffix:
            return True
        else:
            return False

    def isValidDomain(self,domain:str):
        """
        :param domain: str
        :return:
        """
        domain_pattern = "^(?=^.{1,255}$)[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+$"
        ipv4_pattern ="(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
        if re.match(domain_pattern, domain) and (not re.match(ipv4_pattern, domain)):
            return True
        else:
            return False



class Filter():
    def isValidDomain(self,domain:str):
        sf=SingleFilter(100000)
        if (sf.isValidDomain(domain) and sf.haveCorrectTLD(domain) and sf.haveCorrectTLD(domain) and (not sf.inWhiteList(domain))):
            return True
        else:
            return False



if __name__=="__main__":
    pass