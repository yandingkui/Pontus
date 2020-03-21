import redis

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

    ttl_map=dict()

    with open("../result_data/dga_virustotal_A","r") as f:
        for r in f:
            line=r.strip()
            lines=line.split(",")
            domain=lines[3].strip()
            ttl=lines[10].strip()
            ttls=ttl_map.get(domain)
            if ttls is None:
                ttls=set()
                ttls.add(ttl)
                ttl_map[domain]=ttls
            else:
                ttls.add(ttl)
    for k,v in ttl_map.items():
        print("{}:{}".format(k,len(v)))