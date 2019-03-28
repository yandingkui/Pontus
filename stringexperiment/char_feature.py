import sys

sys.path.append("..")
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import pickle
import math
from sklearn.externals.joblib.parallel import Parallel, delayed
import enchant

model_data = pickle.load(open('../source/gib_model.pki', 'rb'))
top100k = pickle.load(open('../source/top_100k.pki', 'rb'))
HEX_DIGITS = set('0123456789abcdef')
VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz')
DIGIT = set("0123456789")

accepted_chars = "abcdefghijklmnopqrstuvwxyz1234567890-_ "
pos = dict([(c, i) for i, c in enumerate(accepted_chars)])

WORDDICT = enchant.Dict("en_US")
DIGIT = set("0123456789")

PINYIN = set(["a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi",
              "bian", "biao", "bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce",
              "cen", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng", "chi", "chong",
              "chou", "chu", "chua", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong",
              "cou", "cu", "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "den",
              "dei", "deng", "di", "dia", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du",
              "duan", "dui", "dun", "duo", "e", "ei", "en", "eng", "er", "fa", "fan", "fang", "fei", "fen",
              "feng", "fo", "fou", "fu", "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng",
              "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha", "hai",
              "han", "hang", "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan",
              "huang", "hui", "hun", "huo", "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing",
              "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang", "kao", "ke", "ken",
              "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la",
              "lai", "lan", "lang", "lao", "le", "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie",
              "lin", "ling", "liu", "long", "lou", "lu", "lv", "luan", "lue", "lve", "lun", "luo", "ma", "mai",
              "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming",
              "miu", "mo", "mou", "mu", "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ng",
              "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu", "nong", "nou", "nu", "nv", "nuan",
              "nve", "nuo", "nun", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi",
              "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu", "qi", "qia", "qian", "qiang", "qiao",
              "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun", "ran", "rang", "rao", "re",
              "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai", "san",
              "sang", "sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shei",
              "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun",
              "shuo", "si", "song", "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai", "tan", "tang",
              "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui",
              "tun", "tuo", "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian",
              "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun", "ya", "yan",
              "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue", "yun", "za",
              "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao",
              "zhe", "zhei", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan",
              "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo"])


def _length(d: str):
    return [len(d)]


def _vowel(d: str):
    num = 0
    consecutive_array = []
    consecutive_num = 0
    for i in d:
        if i in VOWELS:
            num = num + 1
            consecutive_num = consecutive_num + 1
        else:
            if consecutive_num != 0:
                consecutive_array.append(consecutive_num)
                consecutive_num = 0
    if consecutive_num != 0:
        consecutive_array.append(consecutive_num)
    if len(consecutive_array) == 0:
        consecutive_array.append(0)
    max_consecutive_num = np.max(consecutive_array)
    if num == 0:
        ratio = 0
    else:
        ratio = max_consecutive_num / num
    return [num, max_consecutive_num, np.mean(consecutive_array), ratio, num / len(d)]


def _consonant(d: str):
    num = 0
    consecutive_array = []
    consecutive_num = 0
    for i in d:
        if i in CONSONANTS:
            num = num + 1
            consecutive_num = consecutive_num + 1
        else:
            if consecutive_num != 0:
                consecutive_array.append(consecutive_num)
                consecutive_num = 0
    if consecutive_num != 0:
        consecutive_array.append(consecutive_num)
    if len(consecutive_array) == 0:
        consecutive_array.append(0)
    max_consecutive_num = np.max(consecutive_array)
    if num == 0:
        ratio = 0
    else:
        ratio = max_consecutive_num / num
    return [num, max_consecutive_num, np.mean(consecutive_array), ratio,num / len(d)]


def _digit(d: str):
    num = 0
    consecutive_array = []
    consecutive_num = 0
    for i in d:
        if i in DIGIT:
            num = num + 1
            consecutive_num = consecutive_num + 1
        else:
            if consecutive_num != 0:
                consecutive_array.append(consecutive_num)
                consecutive_num = 0
    if consecutive_num != 0:
        consecutive_array.append(consecutive_num)
    if len(consecutive_array) == 0:
        consecutive_array.append(0)
    max_consecutive_num = np.max(consecutive_array)
    if num == 0:
        ratio = 0
    else:
        ratio = max_consecutive_num / num
    return [num, max_consecutive_num, np.mean(consecutive_array), ratio,num / len(d) ]


def _underLine(d: str):
    num = 0
    for i in d:
        if i == "-":
            num = num + 1
    return [num / len(d)]


def _hex(d: str):
    flag = 1
    for i in d:
        if i not in HEX_DIGITS:
            flag = 0
            break
    return [flag]


def _dec(d: str):
    flag = 1
    for i in d:
        if i not in DIGIT:
            flag = 0
            break
    return [flag]


# 1-gram
def _shannon_entropy(d: str):
    m = defaultdict(int)
    for i in d:
        m[i] += 1
    # print(m)
    return [stats.entropy(list(m.values()), base=2)]


def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation,
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]


def ngram(n, l):
    """ Return all n grams from l after normalizing """
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])


def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))


def _readability(d: str):
    model_mat = model_data['mat']
    return [avg_transition_prob(d, model_mat)]


def _top_100k_readability(d: str):
    model_mat = top100k['mat']
    # print("{}".format(np.matrix(model_mat).shape))
    return [avg_transition_prob(d, model_mat)]


def _cover(c):
    if len(c) == 0:
        return [], 0
    c = sorted(c, key=lambda x: (x[0], -x[1]))
    total = 0
    path = []
    L = len(c)
    flags = np.zeros(L)
    for i in range(L):
        if flags[i] == 0:
            tail = []
            for j in range(i + 1, L):
                if c[i][1] <= c[j][0]:
                    tail.append(c[j])
                    flags[j] = 1
            p, t = _cover(tail)
            if (c[i][1] - c[i][0]) + t > total:
                p.append(c[i])
                path = p
                total = (c[i][1] - c[i][0]) + t
        else:
            break
    return sorted(path), total


def _judge_digit(s):
    flag = False
    for i in s:
        if i in DIGIT or i == "-":
            flag = True
            break
    return flag


def _n_gram(d: str, n):
    for start in range(0, len(d) - n + 1):
        if _judge_digit(d[start:start + n]):
            continue
        else:
            yield d[start:start + n], (start, start + n)


def _save_word(d: str):
    result = []
    for L in range(len(d), 1, -1):
        for c, m in _n_gram(d, L):
            if WORDDICT.check(c) or (c in PINYIN):
                result.append((c, m))
    clist = [citem[1] for citem in result]
    r = _cover(clist)
    finalresult = []
    for i in result:
        if i[1] in r[0]:
            finalresult.append(i)
    # print(result)
    return finalresult, r[1], len(result)


def _word_feature(d: str):
    re = _save_word(d)
    max_len = 0
    mean_len = 0
    interval = []
    if re[1] > 0:
        word_order = sorted([o[1] for o in re[0]])
        word_lens = [w[1] - w[0] for w in word_order]
        max_len = np.max(word_lens)
        mean_len = np.mean(word_lens)
        if word_order[0][0]!=0:
            interval.append((0,word_order[0][0]))
        for i in range(len(word_order) - 1):
            if word_order[i][1] != word_order[i + 1][0]:
                interval.append((word_order[i][1], word_order[i + 1][0]))
        if word_order[len(word_order)-1][1] != len(d):
            interval.append((word_order[len(word_order)-1][1],len(d)))
    interval_num_max = 0
    interval_num_min = 0
    interval_num_mean = 0
    all_underline = 0
    if len(interval) > 0:
        interval_num = [I[1] - I[0] for I in interval]
        # print(interval)
        # print(interval_num)
        interval_num_max = np.max(interval_num)
        interval_num_min = np.min(interval_num)
        interval_num_mean = np.mean(interval_num)
        flag = True
        for inter in interval:
            for i in range(inter[0], inter[1]):
                if d[i:i + 1] != "-":
                    flag = False
                    break
        if flag:
            all_underline = 1
    return [ re[2],len(re[0]),re[1] / len(d),  max_len,  mean_len, len(interval), interval_num_max,
            interval_num_min, interval_num_mean, all_underline]


ALL_FEATURES = _length, _vowel, _consonant, _digit, _underLine, _hex,_dec, _shannon_entropy, _readability, _top_100k_readability, _word_feature

def extract_features(d: str, features):
    feature_vector = []
    for f in features:
        try:
            feature_vector = feature_vector + f(d)
        except (ValueError, ArithmeticError) as e:
           print("{} error at {}".format(f,d))
    return feature_vector

# extra all features
def extract_all_features(data, n_jobs=-1):
    """
    Function extracting all available features to a numpy feature array.
    :param data: iterable containing domain name strings
    :return: feature matrix as numpy array
    """
    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    feature_matrix = parallel(
        delayed(extract_features)(d, ALL_FEATURES)
        for d in data
    )
    return np.array(feature_matrix)

def extract_features_2(d,features):
    return (extract_features(d[0], ALL_FEATURES),extract_features(d[1], ALL_FEATURES))

def extract_all_features_for_2(data,n_jobs=-1):
    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    feature_matrix = parallel(
        delayed(extract_features_2)(d, ALL_FEATURES)
        for d in data
    )
    return np.array(feature_matrix)

if __name__ == "__main__":
    # print(extract_features_2(["baidu","qq"],ALL_FEATURES))
    print(len(extract_all_features(["www","s180315349"])[1]))
    #print(extract_features("trlekmynqihxxhy6k",ALL_FEATURES))