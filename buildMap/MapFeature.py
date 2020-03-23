import redis
import numpy as np

def getSingleFeature():
    pass

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


def getFeature(domain):
    redisDomainDB = redis.Redis(host='127.0.0.1', port=6379, db=1)
    redisIPDB = redis.Redis(host='127.0.0.1', port=6379, db=2)
    redisCNAMEDB = redis.Redis(host='127.0.0.1', port=6379, db=3)


    ipset=redisDomainDB.smembers(domain)
    ipgetDomainset=[]
    for ip in ipset:
        ipgetDomainset.append(redisIPDB.hgetall())


    cnameset=redisCNAMEDB.smembers(domain)



if __name__=="__main__":
    pass

