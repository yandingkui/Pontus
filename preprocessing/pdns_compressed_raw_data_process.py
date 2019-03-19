import sys

sys.path.append("..")

from multiprocessing import cpu_count, Process, Queue, Lock
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

pdns_raw_data_dir = pdns_project_dir + '/pdns_gddx_compressed/'

# the data range is a list of [province, date, begin_hour, end_hour]
pdns_raw_data_ranges = [['gdyd', '20180428', 5, 13]]
# package_dir = os.path.join(pdns_project_dir, 'Package')

result_data_dir = '../result_data/20180428'

cpu_number = cpu_count()
thread_number = int(cpu_number/4)


class DataProcessing(Process):

    def __init__(self, file_path_queue, process_index, lock):
        super().__init__()
        self.file_path_queue = file_path_queue
        self.process_index = process_index
        self.lock = lock


    def run(self):
        filter=Filter()
        while (True):
            self.lock.acquire()
            if (self.file_path_queue.empty() == False):
                try:
                    queue_item = self.file_path_queue.get()
                    self.lock.release()
                    ff=str(queue_item[0])
                    result_file_name=ff[ff.rindex("-")+1:ff.index(".txt.bz2")-2]
                    print(result_file_name)
                    hour_result=[]
                    for file_path in queue_item:
                        file_point = bz2.open(file_path, 'r')
                        datalines = file_point.readlines()
                        for line in datalines:
                            try:
                                line = line.decode().strip()
                                linesplit = line.split(',')
                                source_ip = linesplit[0].strip().lower()
                                querydomain = linesplit[3].strip().lower()
                                rcode = linesplit[16].strip()
                                answer = linesplit[19].strip().lower()
                                if filter.isValidDomain(querydomain) and filter.Two_Three_level_domain(querydomain):
                                    hour_result.append(",".join((source_ip,querydomain,rcode,answer)))
                            except:
                                continue
                        file_point.close()

                    with open("/home/yandingkui/dga_detection/result_data/" + result_file_name, mode="w", encoding="utf8") as f:
                        f.write("\n".join(hour_result))
                except:
                    print(traceback.print_exc())

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
            data_province_date_dir = os.path.join(data_province_dir, 'dt=' + date)

            for i in range(begin_hour, end_hour):
                data_province_date_hour_dir = os.path.join(data_province_date_dir, 'hour=' + '%02d' % i)
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
    DataProcessing_process_list = []
    for i in range(0, thread_number):
        DataProcessing_process_list.append(DataProcessing(file_path_queue, i, lock))
        DataProcessing_process_list[i].start()

    for i in range(0, thread_number):
        DataProcessing_process_list[i].join()

    DataFilePathReading_process.terminate()