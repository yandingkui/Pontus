import sys
sys.path.append("..")
# from util.PublicSuffix import PublicSuffix
from util.WhiteLIstUtil import WhiteListUtil
from publicsuffixlist import PublicSuffixList
import re

class SingleFilter():

    def __init__(self,number,psl):
        # self.psl = PublicSuffix().getPublicSuffixList()
        self.psl=psl
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
    def __init__(self):
        self.psl= PublicSuffixList(accept_unknown=False)
        self.sf = SingleFilter(100000, self.psl)

    def isValidDomain(self,domain:str):
        if (self.sf.isValidDomain(domain) and (not self.sf.inWhiteList(domain))):
            return True
        else:
            return False

    def Two_Three_level_domain(self,domain:str):
        """
        identify a domain
        :param domain:  domain:str
        :return: bool
        """
        publicsuffix=self.psl.publicsuffix(domain)
        if publicsuffix==None:
            return False
        pre_domain=domain[:domain.rindex(publicsuffix)-1]
        if len(pre_domain)==0:
            return False
        pre_domain_array=pre_domain.split(".")
        length=len(pre_domain_array)
        if length==2 or length==1:
            return True
        else:
            return False

if __name__=="__main__":
    filter=Filter()
    domains=["www.baidu.com","ddd.adfadfad","9.7.6.5","0=0","qq.com","adfadffadf.com"]
    for d in domains:
        r=filter.Two_Three_level_domain(d)
        print(r)