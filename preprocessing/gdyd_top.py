import sys

sys.path.append("..")

from multiprocessing import Pool
import os
import time

import bz2
from collections import Counter
from publicsuffixlist import PublicSuffixList
from preprocessing.DomainFilter import Filter
import json


class gdyd():
    def __init__(self):
        self.psl = PublicSuffixList(accept_unknown=False)
        self.filter = Filter()

    def statistic_single_hour(self, hour_dir, day, hour:int):
        counter = Counter()
        for minute_file in os.listdir(hour_dir):
            bzfile = os.path.join(hour_dir, minute_file)
            try:
                file_point = bz2.open(bzfile, 'r')
                for line in file_point:
                    try:
                        line = line.decode().strip()
                        linesplit = line.split(',')
                        querydomain = linesplit[3].strip().lower()
                        if self.filter.isValidDomain(querydomain):
                            prisuf = self.psl.privatesuffix(querydomain)
                            if prisuf is not None and prisuf not in self.filter.sf.AleaxTop and \
                                    prisuf not in self.filter.sf.CDNSet and \
                                    prisuf not in self.filter.sf.commonset:
                                counter[prisuf] += 1
                                if prisuf != querydomain:
                                    front = querydomain[:querydomain.rindex(prisuf) - 1]
                                    front_s = front.rsplit(".", 1)
                                    if len(front_s) != 0:
                                        ThreeLD = "{}.{}".format(front_s[len(front_s) - 1], prisuf)
                                        counter[ThreeLD] += 1
                    except:
                        pass
                file_point.close()
            except:
                print("error : {}".format(bzfile))

            print("{} finish".format(bzfile))
        print("{}{} write".format(day, hour))
        with open("../result_data/temp/{}{}.json".format(day, hour), "w") as f:
            f.write(json.dumps(counter))

    def all_day_counter(self, rootpath="/home/public/DNS_Project/pdns_gddx_compressed/gdyd", days=["20180502","20180503","20180504"]):
        s = time.time()
        number = 24
        pool = Pool(number)
        # result = []
        for day in days:
            daydir = os.path.join(rootpath, "dt={}".format(day))
            for h in range(24):
                hourdir = os.path.join(daydir, "hour={0:02d}".format(h))
                if os.path.exists(hourdir):
                    pool.apply_async(func=self.statistic_single_hour, args=(hourdir, day, h,))
                else:
                    print("path error")
                    # result.append(r)
        pool.close()
        pool.join()

        # whole_counter = Counter()
        # for r in result:
        #     whole_counter.update(r.get())
        # for r in whole_counter.most_common(30000):
        #     print("{},{}".format(r[0], r[1]))
        e = time.time()
        print("spend time :{} minutes".format((e - s)/60))


    def get_counter(self,days=["20180502","20180503","20180504"]):
        root_dir="/home/yandingkui/Pontus/result_data/temp/"
        for day in days:
            counter=Counter()
            for i in range(24):
                path=os.path.join(root_dir,"{}{}.json".format(day,i))
                if os.path.exists(path):
                    with open(path, "r") as f:
                        counter1 = Counter(json.loads(f.read()))
                    counter.update(counter1)
            with open("{}{}.json".format(root_dir,day),"w") as f:
                f.write(json.dumps(counter))

    def remove_file(self,days=["20180427","20180428","20180429","20180430","20180501"]):
        root_dir = "/home/yandingkui/Pontus/result_data/temp/"
        for day in days:
            for i in range(24):
                path = os.path.join(root_dir, "{}{}.json".format(day, i))
                if os.path.exists(path):
                    os.remove(path)


    def getBenignDomains(self,days=["20180502","20180503"]):
        root_dir = "/home/yandingkui/Pontus/result_data/temp/"
        for day in days:
            with open(os.path.join(root_dir,"{}.json".format(day)),"r") as f:
                counter=Counter(json.loads(f.read()))
                data=[]
                for item in counter.most_common(30000):
                    data.append(item[0])
                with open("../data_sets/yd_{}".format(day),"w") as F:
                    F.write("\n".join(data))





    def dxvsyd(self,days=["20180427","20171031"]):
        yd = "/home/yandingkui/Pontus/result_data/temp/20180427.json"
        dx="/home/yandingkui/Pontus/result_data/gddx/20171031.json"
        with open(yd,"r") as f:
            counter1 = Counter(json.loads(f.read()))
        with open(dx,"r") as f:
            counter2 = Counter(json.loads(f.read()))
        s1 = []
        s2=[]
        for item in counter1.most_common(30000):
            s1.append(item[0])
        for item in counter2.most_common(30000):
            s2.append(item[0])
        with open("../result_data/yd_20180427","w") as f:
            f.write("\n".join(s1))
        with open("../result_data/dx_20171031","w" ) as f:
            f.write("\n".join(s2))





if __name__ == "__main__":
    gy = gdyd()
    # gy.all_day_counter()
    # gy.get_counter()
    # gy.remove_file(days=["20180502","20180503","20180504"])
    # gy.dxvsyd()
    gy.getBenignDomains()
