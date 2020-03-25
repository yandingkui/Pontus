import sys
sys.path.append("..")
import redis
import numpy as np
import datetime
import publicsuffixlist
from activedomain import DataSetDomains
import traceback
from sklearn.externals.joblib.parallel import Parallel, delayed
from stringexperiment import pontus,comparison
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random
def getDomanListFeature(domain_list):
    parallel = Parallel(n_jobs=-1, verbose=1)
    feature_matrix = parallel(
        delayed(getFeature)(d, datetime.datetime.strptime('20180507', "%Y%m%d"))
        for d in domain_list
    )
    return feature_matrix



def get_ttlmap():
    ttl_map = dict()
    with open("../result_data/dga_virustotal_A", "r") as f:
        for r in f:
            line = r.strip()
            lines = line.split(",")
            domain = lines[3].strip()
            ttl = lines[10].strip()
            ttls = ttl_map.get(domain)
            if ttls is None:
                ttls = set()
                ttls.add(int(ttl))
                ttl_map[domain] = ttls
            else:
                ttls.add(int(ttl))
    return ttl_map


def getFeature(domain,nowdate):
    redisDomainDB = redis.Redis(host='127.0.0.1', port=6379, db=1)
    redisIPDB = redis.Redis(host='127.0.0.1', port=6379, db=2)
    redisCNAMEDB = redis.Redis(host='127.0.0.1', port=6379, db=3)
    offset = datetime.timedelta(days=7)
    beforeWeekDate=nowdate-offset

    psl = publicsuffixlist.PublicSuffixList(accept_unknown=False)

    vector=np.zeros(11)
    ipset=redisDomainDB.smembers(domain)

    alldays = []
    weekdays = []
    priratios = []

    if (ipset is not None) and len(ipset)>0:
        # 域名解析IP的个数
        vector[0] = len(ipset)


        #ip承载的
        for ip in ipset:
            #统计所有的域名数量
            ip_domain_map=redisIPDB.hgetall(ip)
            ipAllNum=len(ip_domain_map)
            alldays.append(ipAllNum)
            pri_map=dict()

            weeknum=0

            for k,v in ip_domain_map.items():
                domain_time_str=str(v)
                vs=domain_time_str[domain_time_str.index("'")+1:domain_time_str.rindex("'")]
                domain_name_str=str(k)
                dns = domain_name_str[domain_name_str.index("'") + 1:domain_name_str.rindex("'")]

                domain_time=datetime.datetime.strptime(vs,'%Y%m%d%H%M%S')
                #如果该域名在一个星期之内
                if domain_time>beforeWeekDate:
                    weeknum=weeknum+1
                #统计子域名个数
                domain_pri=psl.privatesuffix(dns)
                pri_num=pri_map.get(domain_pri)
                if pri_num is None:
                    pri_map[domain_pri]=1
                else:
                    pri_map[domain_pri]=pri_num+1

            weekdays.append(weeknum)
            priratios.append(max(pri_map.values())/ipAllNum)


    #解析IP一周内的解析情况
    if len(alldays)>0:
        vector[1]=np.max(alldays)
        vector[2]=np.min(alldays)
        vector[3]=np.mean(alldays)
    #
    if len(weekdays)>0:
        vector[4]=np.max(weekdays)
        vector[5]=np.min(weekdays)
        vector[6]=np.mean(weekdays)

    if len(priratios)>0:
        vector[7] = np.max(priratios)
        vector[8] = np.min(priratios)
        vector[9] = np.mean(priratios)


    cnameset=redisCNAMEDB.smembers(domain)
    if cnameset is not None :
        vector[10]=1

    return vector


def xgboost_test(train_x, test_x, train_y, test_y):
    # iris = datasets.load_iris()
    # data = iris.data[:100]
    #
    #
    # # 一共有100个样本数据, 维度为4维
    # label = iris.target[:100]
    #
    #
    # train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 12,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1}

    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=10, evals=watchlist)
    # 输出概率
    ypred = bst.predict(dtest)

    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    y_pred = (ypred > 0.5) * 1

    from sklearn import metrics
    print("xgboost")
    print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))
    print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
    print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
    print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
    print(metrics.confusion_matrix(test_y, y_pred))


if __name__=="__main__":

    trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = DataSetDomains.getDomains()
    print(len(trainDGADomain))
    print(len(testDGADomain))
    print(len(trainBenignDomain))
    print(len(testBenignDomain))

    trainDomains = trainDGADomain + trainBenignDomain

    trainLabel_noshuffle = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))

    testDomains = testDGADomain + testBenignDomain
    testLabel = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

    ppp=pontus.pontus()
    str_train_features = ppp.getDomainFeatures(trainDomains)
    map_train_features=getDomanListFeature(trainDomains)


    train_features_noshuffle=np.append(str_train_features, map_train_features, axis=1)
    print(str_train_features[0])
    print(map_train_features[0])
    print(train_features_noshuffle[0])

    index=[i for i in range(len(trainDomains))]
    random.shuffle(index)
    train_features=[train_features_noshuffle[i] for i in index]
    trainLabel=[trainLabel_noshuffle[i] for i in index]

    clf = GradientBoostingClassifier(max_depth=24, n_estimators=260, max_features=36)
    # # clf=RandomForestClassifier(n_estimators=755, max_features=28, criterion='gini')
    clf.fit(train_features, trainLabel)

    str_pred_features = ppp.getDomainFeatures(testDomains)
    map_pred_features=getDomanListFeature(testDomains)
    pre_features = np.append(str_pred_features, map_pred_features, axis=1)

    xgboost_test(train_features,pre_features,trainLabel,testLabel)
    #
    predict_result = clf.predict(pre_features)

    print("GBDT")
    ppp.printMetric(testLabel,predict_result)