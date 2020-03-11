import sys

sys.path.append("..")

from multiprocessing import Pool,Manager
import os
import time
import json
import bz2
from collections import Counter
from preprocessing.DomainFilter import Filter
from publicsuffixlist import PublicSuffixList
#
# def statistic_single_day(day,rootpath="/home/public/DNS_Project/gddx"):
#     filter=Filter()
#     flag_dict=dict()
#     counter =Counter()
#     daydir=os.path.join(rootpath,day)
#     if os.path.exists(daydir):
#         files=os.listdir(daydir)
#         for hour in files:
#             hour_dir=os.path.join(daydir,hour)
#             for minute_file in os.listdir(hour_dir):
#                 bzfile=os.path.join(hour_dir,minute_file)
#                 file_point = bz2.open(bzfile, 'r')
#                 for line in file_point:
#                     try:
#                         line = line.decode().strip()
#                         linesplit = line.split(',')
#                         querydomain = linesplit[3].strip().lower()
#                         flag=flag_dict.get(querydomain)
#                         if flag is None:
#                             if filter.Two_Three_level_domain(querydomain) and filter.isValidDomain(querydomain) and (not filter.inWhiteList(querydomain)):
#                                 counter[querydomain]+=1
#                                 flag_dict[querydomain] = True
#                             else:
#                                 flag_dict[querydomain] = False
#                         elif flag == True:
#                             counter[querydomain] += 1
#                         else:
#                             continue
#                     except:
#                         print("except")
#                 print("{} finish".format(bzfile))
#
#     return counter
#
#
# def all_day_counter(days):
#     s=time.time()
#     number=len(days)
#     pool=Pool(number)
#     result=[]
#     for d in days:
#         r=pool.apply_async(func=statistic_single_day,args=(d,))
#         result.append(r)
#     pool.close()
#     pool.join()
#
#     whole_counter=Counter()
#     for r in result:
#         whole_counter.update(r.get())
#     for r in whole_counter.most_common(30000):
#         print("{},{}".format(r[0],r[1]))
#     e=time.time()
#     print("spend time :{}".format(e-s))
#
#
# if __name__=="__main__":
#     all_day_counter(['20171102','20171105','20171101','20171104','20171106','20171031','20171103'])
#     # all_day_counter(['20171102'])



class gddx():
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
        with open("../result_data/gddx/{}{}.json".format(day, hour), "w") as f:
            f.write(json.dumps(counter))

    def all_day_counter(self, rootpath="/home/public/DNS_Project/gddx", days=["20171031"]):
        s = time.time()
        number = 36
        pool = Pool(number)
        # result = []
        for day in days:
            daydir = os.path.join(rootpath, "{}".format(day))
            for h in range(24):
                hourdir = os.path.join(daydir, "hour={0:02d}".format(h))
                if os.path.exists(hourdir):
                    pool.apply_async(func=self.statistic_single_hour, args=(hourdir, day, h,))
                else:
                    print("{}   path error".format(hourdir))
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

    def get_counter(self,days = ["20171101", "20171102", "20171103", "20171104", "20171105", "20171106"]):
        root_dir="/home/yandingkui/Pontus/result_data/gddx/"
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

    def remove_file(self,days = ["20171101", "20171102", "20171103", "20171104", "20171105", "20171106"]):
        root_dir="/home/yandingkui/Pontus/result_data/gddx/"
        for day in days:
            for i in range(24):
                path = os.path.join(root_dir, "{}{}.json".format(day, i))
                if os.path.exists(path):
                    os.remove(path)

if __name__ == "__main__":
    days = ["20171101", "20171102", "20171103", "20171104", "20171105", "20171106"]
    gy = gddx()
    # gy.all_day_counter(days = ["20171101", "20171102", "20171103", "20171104", "20171105", "20171106"])
    # gy.get_counter(days)
    gy.remove_file(days = ["20171101", "20171102", "20171103", "20171104", "20171105", "20171106"])
