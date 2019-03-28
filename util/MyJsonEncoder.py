import json
import numpy as np
import datetime


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj,datetime.date):
            return obj.strftime(obj,'%Y-%m-%d')
        elif isinstance(obj,set):
            return list(obj)
        else:
            try:
                return json.JSONEncoder.default(self, obj)
            except:
                return obj.__str__()
