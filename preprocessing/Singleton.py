import sys
sys.path.append("..")

import threading
class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"):
                    Singleton._instance = object.__new__(cls)
        return Singleton._instance


def task(arg):
    obj = Singleton()
    obj.hello(arg)

for i in range(10):
    t = threading.Thread(target=task,args=(i,))
    t.start()