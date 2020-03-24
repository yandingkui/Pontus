import redis
import numpy as np
import datetime
import publicsuffixlist


def getDomanListFeature(domain_list):
    ttl_map=get_ttlmap()

    for domain in domain_list:
        dfea=np.zeros(13)



def get_ttlmap():
    ttl_map = dict()
    with open("../result_data/dga_virustotal_A", "r") as f:
        for r in f:
            line = r.strip()
            lines = line.split(",")
            domain = lines[3].strip()
            ttl = lines[10].strip()
            ttls = ttl_map.get(domain)
            if ttls is None:
                ttls = set()
                ttls.add(int(ttl))
                ttl_map[domain] = ttls
            else:
                ttls.add(int(ttl))
    return ttl_map


def getFeature(domain,nowdate):
    redisDomainDB = redis.Redis(host='127.0.0.1', port=6379, db=1)
    redisIPDB = redis.Redis(host='127.0.0.1', port=6379, db=2)
    redisCNAMEDB = redis.Redis(host='127.0.0.1', port=6379, db=3)
    offset = datetime.timedelta(days=7)
    beforeWeekDate=nowdate-offset
    psl = publicsuffixlist.PublicSuffixList(accept_unknown=False)

    vector=np.zeros(11)
    ipset=redisDomainDB.smembers(domain)
    #域名解析IP的个数
    vector[0]=len(ipset)

    alldays=[]
    weekdays=[]
    priratios=[]

    #ip承载的
    for ip in ipset:
        #统计所有的域名数量
        ip_domain_map=redisIPDB.hgetall(ip)
        ipAllNum=len(ip_domain_map)
        alldays.append(ipAllNum)
        pri_map=dict()

        weeknum=0

        for k,v in ip_domain_map:
            domain_time=datetime.datetime.strptime(v,'%Y%m%d%H%M%S')
            if domain_time>beforeWeekDate:
                weeknum=weeknum+1
            domain_pri=psl.privatesuffix(k)

            pri_num=pri_map.get(domain_pri)
            if pri_num is None:
                pri_map[domain_pri]=1
            else:
                pri_num[domain_pri]=pri_num+1

        weekdays.append(weeknum)
        priratios.append(max(pri_map.values())/ipAllNum)


    #解析IP一周内的解析情况
    if len(alldays)>0:
        vector[1]=np.max(alldays)
        vector[2]=np.min(alldays)
        vector[3]=np.mean(alldays)
    #
    if len(weekdays)>0:
        vector[4]=np.max(weekdays)
        vector[5]=np.min(weekdays)
        vector[6]=np.mean(weekdays)

    if len(priratios)>0:
        vector[7] = np.max(priratios)
        vector[8] = np.min(priratios)
        vector[9] = np.mean(priratios)


    cnameset=redisCNAMEDB.smembers(domain)
    if cnameset is not None :
        vector[10]=1



if __name__=="__main__":
    getFeature("www.baidu.com",datetime.datetime.strptime('20180507','%Y%m%d'))


