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
from multiprocessing import Pool

class labelclassifier():

    def __init__(self):
        pass

    def get_benign_labels(self):
        pass

    def get_AGD_labels(self):
        pass

