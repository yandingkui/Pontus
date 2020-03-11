import sys

sys.path.append("..")

from multiprocessing import cpu_count, Process, Queue, Lock
from MatchDGA import Match
import os
import time
import bz2
import json
import re
from util import MyJsonEncoder
from preprocessing.DomainFilter import Filter
import traceback
pdns_project_dir = os.path.abspath("..")
pdns_project_dir=os.path.abspath("/home/public/DNS_Project/")

# pdns_raw_data_dir = pdns_project_dir + '/pdns_gddx_compressed/'
pdns_raw_data_dir = pdns_project_dir
# the data range is a list of [province, date, begin_hour, end_hour]
pdns_raw_data_ranges = [['gddx', '20171031', 0, 23]]
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
                    for line in file_point:
                        try:
                            line = line.decode().strip()
                            linesplit = line.split(',')
                            querydomain = linesplit[3].strip().lower()
                            if(filter.isValidDomain(querydomain) and self.match.judge(querydomain)):
                                self.match.saveFile(line)
                                print(querydomain)

                        except:
                            print("error info:{}\n file:{}".format(traceback.print_exc(),file_path))
                    file_point.close()
                day_string=result_file_name[:len(result_file_name)-2]


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