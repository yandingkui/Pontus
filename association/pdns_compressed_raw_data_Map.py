import sys

sys.path.append("..")

from multiprocessing import cpu_count, Process, Queue, Lock
from MatchDGA import Match
import os
import time
import bz2
import re
import json
import re
from util import MyJsonEncoder
import redis
from preprocessing.DomainFilter import Filter
import traceback
pdns_project_dir = os.path.abspath("..")
pdns_project_dir=os.path.abspath("/media/mnt/pdns_gddx_compressed/")

# pdns_raw_data_dir = pdns_project_dir + '/pdns_gddx_compressed/'
pdns_raw_data_dir = pdns_project_dir
# the data range is a list of [province, date, begin_hour, end_hour]
pdns_raw_data_ranges = [['gdyd', '20180321', 0, 23]]
# pdns_raw_data_ranges = [['gdyd', '20180321', 0, 23],
#                         ['gdyd', '20180322', 0, 23],
#                         ['gdyd', '20180323', 0, 23],
#                         ['gdyd', '20180324', 0, 23],
#                         ['gdyd', '20180422', 0, 23],
#                         ['gdyd', '20180423', 0, 23],
#                         ['gdyd', '20180424', 0, 23],
#                         ['gdyd', '20180425', 0, 23],
#                         ['gdyd', '20180426', 0, 23],
#                         ['gdyd', '20180427', 0, 23],
#                         ['gdyd', '20180428', 0, 23],
#                         ['gdyd', '20180429', 0, 23],
#                         ['gdyd', '20180430', 0, 23],
#                         ['gdyd', '20180501', 0, 23],
#                         ['gdyd', '20180502', 0, 23],
#                         ['gdyd', '20180503', 0, 23],
#                         ['gdyd', '20180504', 0, 23],
#                         ['gdyd', '20180505', 0, 23],
#                         ['gdyd', '20180506', 0, 23],
#                         ['gdyd', '20180507', 0, 23],
#                         ['gdyd', '20180508', 0, 23],
#                         ['gdyd', '20180509', 0, 23]]
# package_dir = os.path.join(pdns_project_dir, 'Package')


cpu_number = cpu_count()
thread_number = int(cpu_number)


class DataProcessing(Process):

    def __init__(self, file_path_queue, process_index, lock,match):
        super().__init__()
        self.file_path_queue = file_path_queue
        self.process_index = process_index
        self.lock = lock
        self.match=match


    def run(self):
        filter=Filter()
        while (True):
            self.lock.acquire()
            if (self.file_path_queue.empty() == False):
                queue_item = self.file_path_queue.get()
                self.lock.release()
                ff=str(queue_item[0])
                if ff.__contains__("-"):
                    result_file_name=ff[ff.rindex("-")+1:ff.index(".txt.bz2")-2]
                else:
                    result_file_name = ff[:ff.index(".txt.bz2") - 2]
                print(result_file_name)

                for file_path in queue_item:

                    file_point = bz2.open(file_path, 'r')
                    answerset = set()
                    redis_domain=redis.Redis(host='127.0.0.1',port=6379,db=1)
                    redis_ip=redis.Redis(host='127.0.0.1',port=6379,db=2)
                    redis_CNAME = redis.Redis(host='127.0.0.1', port=6379, db=3)
                    for line in file_point:
                        try:
                            line = line.decode().strip()
                            linesplit = line.split(',')
                            querydomain = linesplit[3].strip().lower()
                            type=linesplit[4].strip()
                            time=linesplit[11].strip()
                            isMULL=linesplit[14].strip()
                            answer =linesplit[18].strip().lower()
                            keys=",".join((querydomain,answer))
                            if(type=='A' and  isMULL != 'MULL' and len(answer)>0):
                                if answerset.__contains__(keys):
                                    continue
                                else:
                                    ips=answer.split(";")
                                    ipv4_pattern = "(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
                                    for ip in ips:
                                        if re.match(ipv4_pattern,ip):
                                            redis_domain.sadd(querydomain,ip)
                                            redis_ip.hset(ip,querydomain,time)
                                        else:
                                            redis_CNAME.sadd(querydomain,ip)
                        except:
                            print("error info:{}\n file:{}".format(traceback.print_exc(),file_path))
                    file_point.close()
                    print(result_file_name+"  finish hahhahah")
                    redis_domain.close()
                    redis_ip.close()
                    redis_CNAME.close()
            else:
                self.lock.release()
                break
        print('Processing ' + str(self.process_index) + ' is finished')


class DataFilePathReading(Process):
    def __init__(self, pdns_raw_data_dir, pdns_raw_data_ranges, file_path_queue):
        super().__init__()
        self.pdns_raw_data_dir = pdns_raw_data_dir
        self.pdns_raw_data_ranges = pdns_raw_data_ranges
        self.file_path_queue = file_path_queue

    def run(self):
        for data_range in pdns_raw_data_ranges:
            province = data_range[0]
            date = data_range[1]
            begin_hour = data_range[2]
            end_hour = data_range[3]

            data_province_dir = os.path.join(self.pdns_raw_data_dir, province)
            if province=="gdyd":
                data_province_date_dir = os.path.join(data_province_dir, 'dt=' + date)
            else:
                data_province_date_dir = os.path.join(data_province_dir, date)

            for i in range(begin_hour, end_hour+1):
                data_province_date_hour_dir = os.path.join(data_province_date_dir, 'hour=' + '%02d' % i)
                if os.path.exists(data_province_date_hour_dir) == False:
                    print(data_province_date_hour_dir)
                    continue
                filenames = os.listdir(data_province_date_hour_dir)
                item=[]
                for filename in filenames:
                    file_path = os.path.join(data_province_date_hour_dir, filename)
                    item.append(file_path)
                self.file_path_queue.put(item)

if __name__ == '__main__':
    file_path_queue = Queue()
    DataFilePathReading_process = DataFilePathReading(pdns_raw_data_dir, pdns_raw_data_ranges, file_path_queue)
    DataFilePathReading_process.start()
    time.sleep(1)
    lock = Lock()
    match = Match.Match()
    DataProcessing_process_list = []
    for i in range(0, thread_number):
        DataProcessing_process_list.append(DataProcessing(file_path_queue, i, lock,match))
        DataProcessing_process_list[i].start()

    for i in range(0, thread_number):
        DataProcessing_process_list[i].join()

    DataFilePathReading_process.terminate()