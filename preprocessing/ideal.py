import os
import sys

sys.path.append("..")
import requests
import json
import re
import numpy as np
import pandas as pd
from publicsuffixlist import PublicSuffixList
from stringexperiment.char_feature import extract_all_features
from sklearn.externals import joblib
from datetime import datetime
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
def statistic(days):
    counter = Counter()

    for day in days:
        for n in range(24):
            filepath = os.path.join("../result_data/{}".format(day), "{}{}".format(day, n))
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for r in f:
                        rsplit = r.strip().split(",")
                        d = rsplit[1]
                        counter[d]+=1
    for a in counter.most_common(30000):
        print(a)


if __name__ == "__main__":
    statistic(["20180427","20180428","20180429","20180430","20180501","20180502","20180503","20180504"])
