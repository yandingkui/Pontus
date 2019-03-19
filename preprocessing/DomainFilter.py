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
        if (sf.isValidDomain(domain) and sf.haveCorrectTLD(domain)  and (not sf.inWhiteList(domain))):
            return True
        else:
            return False

    def Two_Three_level_domain(self,domain:str):
        psl = PublicSuffix().getPublicSuffixList()
        publicsuffix=psl.publicsuffix(domain)
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


    def handle(self, datalines, hour_result,ip_set):
        # filter = Filter()
        # for line in datalines:
        #     try:
        #         line = line.decode().strip()
        #         linesplit = line.split(',')
        #         source_ip = linesplit[0].strip().lower()
        #         querydomain = linesplit[3].strip().lower()
        #         rcode = linesplit[16]
        #         ds = querydomain.split(".")
        #         if len(ds) == 2 or len(ds) == 3:
        #             status = filter.get_nx_ac_domains(querydomain, rcode)
        #             if status == 0:
        #                 continue
        #             else:
        #                 ac_nx_set = hour_dict.get(source_ip)
        #                 if not ac_nx_set:
        #                     ac_nx_set = [set(), set()]
        #                 if status == 2:
        #                     ac_nx_set[1].add(querydomain)
        #                 elif status == 1:
        #                     ac_nx_set[0].add(querydomain)
        #                 hour_dict[source_ip] = ac_nx_set
        #     except:
        #         continue
        ipv4_pattern = re.compile("(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])")

        for line in datalines:
            try:
                line = line.decode().strip()
                linesplit = line.split(',')
                source_ip = linesplit[0].strip().lower()
                querydomain = linesplit[3].strip().lower()
                rcode = linesplit[16].strip()
                answer=linesplit[19].strip().lower()
                if rcode != "3" and len(answer)!=0:
                    answer_split=answer.split(";")
                    for a in answer_split:
                        if ipv4_pattern.match(a) and a in ip_set:
                            hour_result.append(line)
            except:
                continue

if __name__=="__main__":
    filter=Filter()
    domains=["www.baidu.com","ddd.adfadfad","9.7.6.5","0=0","qq.com","adfadffadf.com"]
    for d in domains:
        r=filter.Two_Three_level_domain(d)
        print(r)