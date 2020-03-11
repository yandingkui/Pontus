import sys

sys.path.append("..")
import re
import publicsuffixlist


class RegexMatch():
    RegexList = ['[a-z]{6,11}\.(dyndns.org|yi.org|dynserv.com|mooo.com)$',
                 '^ns1\.[a-z]{5,13}[0-9]{1,2}\.(biz|com|info|net|org)$',
                 '[a-z]{16}\.(ru)$',
                 '[a-z0-9]{12,18}\.com$',
                 '([a-z]{7,30})\.com$',
                 '[0-9a-f]{32}\.(org|info|co\.cc|cz.cc)$',
                 '[a-y]{12,15}\.(net|biz|org|com|ru|info|co.uk)$',
                 '[a-f0-9]{2,11}\.(com)$',
                 '[a-y0-8]{12,23}\.(cc|cn|com|ddns.net|in|sg|ws)$',
                 '[a-z]{5,11}\.(biz|cc|cn|com|info|net|org|ws)$',
                 '[a-z0-9]{12,24}\.(biz|cn|com|info|net|org|ru)$',
                 '[0-9a-f]{16}\.(cn|net)$',
                 '[\s\S]{6}\.com$',
                 '[a-z0-9]{34}\.(cc|ws|to|in|hk|cn|tk|so)$',
                 '[a-z]{10}\.com$',
                 '[a-z0-9]{7}\.(com|org|net|info)$',
                 '[a-z]{8,20}\.(com)$',
                 '[a-y]{16}\.(eu)$',
                 '[a-f0-9]{1,8}\.(eu)$',
                 '([a-z]{16}|[a-z]{18})\.(ru)$',
                 '[a-z]{10}\.com$|[a-z]{17}\.net$',
                 '[a-z]{12,24}\.(at|biz|club|cn|com|date|eu|in|info|load|me|mn|net|org|pro|pw|ru|su|tk|top|ua|win|xyz)$',
                 '[a-z]{5,11}\.(com)$',
                 '[a-z]{11,32}\.(net|biz|org|com|info|ru)$',
                 '[a-z0-9]{20,28}\.(net|biz|org|com)$',
                 '[0-9a-f]{16}\.(net)$',
                 '[a-z]{8,24}\.com$',
                 '[a-f0-9]{8}\.(net|space|top)$',
                 '[a-y]{5,18}\.(be|biz|click|de|eu|fr|in|info|it|nl|org|pl|pm|pw|ru|su|tf|uk|us|work|xyz|yt)$',
                 '[a-z0-9]{8,11}\.(ru)$',
                 '[a-z-]{12,24}\.com$',
                 '[a-z]{8,16}\.(com|net|org|info|biz)$',
                 '([a-y]{12})\.(online|tech|support)$',
                 '(www\.){0,1}[a-z0-9]{10}\.(com|org|info|net)$',
                 '[a-z]{5,12}\.(ru|net|org|in|com|biz|info|net|pw|xyz)$',
                 '[a-y]{7,26}\.(cx|mu|in|ms|im|ki|mx|tv|ir|me|cm|to|co|mn|kz|cc|sx|pw|ru|ws|de|tw|pro|sh|bz|bit|so|nf|sc|biz|us|la|jp|com|ug|nu|su|ac|ga|net|tj|org|xxx|eu)$',
                 '[0-9a-f]{32}\.(net)$',
                 '[a-z]{7,12}\.(cc|com|dyndns.org|net|tv)$',
                 '[a-z]{10}\.(com|net|org|ru|tv)$',
                 '[0-9a-h]{7,8}\.(com|net)$',
                 '[a-z]{6,15}\.(com|biz|net|org|info|cc)$',
                 '[a-z]{6,12}\.(biz|cc|co|com|eu|in|info|name|net|org|ru|se)$',
                 '[a-z]{6,12}\.(com|net|org|info)$',
                 '[a-z]{8,12}\.(kz|com)$',
                 '[abcdefnfolmk]{16}\.(cc|co|co.uk|com|de|eu|ga|info|net|online|org|tk|website)$',
                 '[a-z]{8,25}\.(com|net|org|info|biz)$',
                 '[a-z0-9]{12}\.(com|org|net|top)$',
                 '([a-y]{14}|[a-y]{17})\.(in|me|cc|su|tw|net|com|pw|org)$',
                 '[acegikmoqsuwy]{16}\.(org)$',
                 '[a-z\-]{9,15}\.(net)$',
                 '[a-z1-8]{18}\.(com|ru|net|biz|cn)$',
                 '[a-y]{8,19}\.(com|click|bid|eu)$',
                 '[aeioubcdfghklmnpqrstvwxz]{8,15}\.ddns.net$',
                 '[a-z]{7,30}\.(net|ru)$',
                 '[m][djtz][acegikmquy][wx][mno][djtz][i][wx][mno][djtz][acegikmquy][a]\.(com|org|net|info)$',
                 '[a-z]{16}\.(com)$',
                 '[qwertyuiopasdfg]{8}\.com$',
                 '[a-y]{7}\.(info|eu)$',
                 '[a-z]{16}\.(info|mynumber.org|ru)$',
                 '([qwrtpsdfghjklzxcvbnm][eyuioa])+([qwrtpsdfghjklzxcvbnm]{0,1})\.(eu|info|com|su|net)$',
                 '0-0-0-0-0-0-0-0-0-0-0-0-0-[0-9]{1,2}-0-0-0-0-0-0-0-0-0-0-0-0-0\.(info)$',
                 '[a-z]{7,11}\.(com|info|net|org)$',
                 '[a-z]{7,11}\.(com|info|net|org)$',
                 '[a-y]{12}\..{2,7}$',
                 '[a-y][a-j][a-k]?[a-y]{1,2}([a-y]|[1-9])(anj|ebf|arm|pra|aym|unj|ulj|uag|esp|kot|onv|edc|naj|bef|ram|rpa|yam|nuj|luj|gau|pes|tko|von|ced)\.(biz|com|net)$',
                 '[a-z]{7,13}\.(biz|ch)$',
                 '[qwertyuiopasdfghjklzxcvbnm123945]{10,15}\.(com|net)$',
                 '(sn|al)[a-z]{6}\.(com|in|net|org|ru)$',
                 '[redotnxpl]{14}\.info$|[flashpyergtdob]{19}\.net$',
                 '[a-z]{7,13}\.(com|dyndns.org|net)$',
                 '[a-z]{6}\.com$',
                 '[a-z]{7,11}\.(com|ru|top)$',
                 '(wd)?[0-9a-f]{32}\.(pro|win)$',
                 '[xx[0-9a-f]{0,8}\.(ac|ag|am|at|be|bz|cc|ch|cn|cz|de|dk|es|eu|fm|fr|gr|hk|im|in|io|it|kz|la|li|lv|md|me|ms|nu|pl|ru|sc|se|sg|sh|su|tc|tk|tm|tv|tw|us|ws)$']

    @classmethod
    def match(cls, domain):
        pr = publicsuffixlist.PublicSuffixList()
        prd=pr.privatesuffix(domain)
        if prd == None:
            prd=""
        flag = False
        for p in cls.RegexList:
            if re.match(p, domain) !=None or re.match(p, prd) != None:
                flag = True
                break
        return flag

if __name__=="__main__":
    domain_set=set()
    with open("/home/yandingkui/Pontus/result_data/20180427/shodongbiaoji","r") as f:
        for d in f:
            if not RegexMatch.match(d.strip()):
                domain_set.add(d)
                print(d)
    with open("../result_data/benign","w") as f:
        f.write("\n".join(domain_set))
