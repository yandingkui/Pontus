import redis

def getSingleFeature(domain):
    redis_domain = redis.Redis(host='127.0.0.1', port=6379, db=1)
    redis_ip = redis.Redis(host='127.0.0.1', port=6379, db=2)
    redis_CNAME = redis.Redis(host='127.0.0.1', port=6379, db=3)

