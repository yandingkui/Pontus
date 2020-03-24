import sys
sys.path.append("..")

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
from preprocessing.DomainFilter import Filter
import os
import bz2
import json
import random


def xmeans_model(sample):
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    max_num = int(len(sample) / 2)
    xmeans_instance = xmeans(sample, initial_centers, max_num)
    xmeans_instance.process()
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()


    return clusters,centers

def run(hourPath):
    filter = Filter()
    pathMap=dict()
    for l in os.listdir(hourPath):
        #获得每5分钟的数据文件
        timeStr=l[l.rindex("-")+1:l.rindex(".txt.bz")]
        fiveMinFiles=pathMap.get(timeStr)
        if fiveMinFiles is None:
            fiveMinFiles=[]
            fiveMinFiles.append(l)
            pathMap[timeStr]=fiveMinFiles
        else:
            fiveMinFiles.append(l)
        # 读取数据
    print("get all file names")

    for k,v in pathMap.items():
        total=0
        ttl_map=dict()
        visit_map=dict()
        nx_map=dict()
        for bzf in v:
            file_point = bz2.open("{}/{}".format(hourPath,bzf), 'r')

            for line in file_point:
                total=total+1
                try:
                    line = line.decode().strip()
                    linesplit = line.split(',')
                    querydomain = linesplit[3].strip().lower()
                    if (filter.isValidDomain(querydomain) and filter.Two_Three_level_domain(querydomain)):

                        visitIP = linesplit[0].strip()
                        ttl=linesplit[10].strip()
                        isMULL = linesplit[15].strip()
                        # 存储访问关系
                        visitList=visit_map.get(querydomain)
                        if visitList is None:
                            visitList=set()
                            visitList.add(visitIP)
                            visit_map[querydomain]=visitList
                        else:
                            visitList.add(visitIP)

                        if isMULL == "MULL":
                            nxDomainList=nx_map.get(visitIP)
                            if nxDomainList is None:
                                nxDomainList=set()
                                nxDomainList.add(querydomain)
                                nx_map[visitIP]=nxDomainList
                            else:
                                nxDomainList.add(querydomain)
                        else:
                            # ttl
                            ttlList = ttl_map.get(querydomain)
                            if ttlList is None:
                                ttlList = set()
                                ttlList.add(ttl)
                                ttl_map[querydomain] = ttlList
                            else:
                                ttlList.add(ttl)
                except:
                    continue

            print("{}  read finish".format(bzf))
        print("total:{}".format(total))
        print("all domains:{}".format(len(visit_map)))
        print("all active :{}".format(len(ttl_map)))
        print("all NX:{}".format(len(nx_map)))




def testActiveDomain(days):
    rootdir="/media/mnt/pdns_gddx_compressed/gdyd/dt="
    for d in days:
        dayPath=rootdir+d
        for i in range(24):
            hourPath="{}/{}".format(dayPath,"hour=%02d"%i)
            run(hourPath)


def testXGboost():


    train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)
    import xgboost as xgb
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1}

    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=5, evals=watchlist)
    # 输出概率
    ypred = bst.predict(dtest)

    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    y_pred = (ypred >= 0.5) * 1

    from sklearn import metrics
    print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))
    print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
    print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
    print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
    print(metrics.confusion_matrix(test_y, y_pred))


if __name__=="__main__":
    # map=dict()
    # map['20180730']=1
    # map['20180976']=2
    # map['20170000']=3
    # for k in sorted(map.keys()):
    #     print(map.get(k))
    index=[i for i in range(4)]
    random.shuffle(index)
    print(index)
