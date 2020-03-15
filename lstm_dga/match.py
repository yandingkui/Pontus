import json
import numpy as np
from publicsuffixlist import PublicSuffixList
class CutWords:
    order = dict()

    def __init__(self):

        dict_path = '../source/dict.txt'
        # self.word_dict = self.load_words(dict_path)

        str = "abcdefghijklmnopqrstuvwxyz1234567890-_"
        i = 1
        for c in str:
            CutWords.order[c] = i
            i = i + 1
        with open(dict_path, "r") as fd:
            for r in fd:
                r = r.strip()
                if r in CutWords.order.keys():
                    continue
                else:
                    CutWords.order[r.strip()] = i
                    i = i + 1
        self.word_dict=CutWords.order.keys()

    # # 加载词典
    # def load_words(self, dict_path):
    #     words = list()
    #     for line in open(dict_path):
    #         words += line.strip().split(' ')
    #     return words



    # 最大向前匹配
    def max_forward_cut(self, sent):
        # 1.从左向右取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。
        # 2.查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。
        cutlist = []
        index = 0
        max_wordlen = 5
        while index < len(sent):
            matched = False
            for i in range(max_wordlen, 0, -1):
                cand_word = sent[index: index + i]
                if cand_word in self.word_dict:
                    cutlist.append(cand_word)
                    matched = True
                    break

            # 如果没有匹配上，则按字符切分
            if not matched:
                i = 1
                cutlist.append(sent[index])
            index += i
        return cutlist

    # 最大向后匹配
    def max_backward_cut(self, sent):
        # 1.从右向左取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。
        # 2.查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。
        cutlist = []
        index = len(sent)
        max_wordlen = 5
        while index > 0:
            matched = False
            for i in range(max_wordlen, 0, -1):
                tmp = (i + 1)
                cand_word = sent[index - tmp: index]
                # 如果匹配上，则将字典中的字符加入到切分字符中
                if cand_word in self.word_dict:
                    cutlist.append(cand_word)
                    matched = True
                    break
            # 如果没有匹配上，则按字符切分
            if not matched:
                tmp = 1
                cutlist.append(sent[index - 1])

            index -= tmp

        return cutlist[::-1]

    # 双向最大向前匹配
    def max_biward_cut(self, sent):
        # 双向最大匹配法是将正向最大匹配法得到的分词结果和逆向最大匹配法的到的结果进行比较，从而决定正确的分词方法。
        # 启发式规则：
        # 1.如果正反向分词结果词数不同，则取分词数量较少的那个。
        # 2.如果分词结果词数相同 a.分词结果相同，就说明没有歧义，可返回任意一个。 b.分词结果不同，返回其中单字较少的那个。
        forward_cutlist = self.max_forward_cut(sent)
        backward_cutlist = self.max_backward_cut(sent)
        count_forward = len(forward_cutlist)
        count_backward = len(backward_cutlist)

        def compute_single(word_list):
            lens=[len(i) for i in word_list]
            # print("方差：{}".format(np.var(lens)))
            return np.var(lens)

        if count_forward == count_backward:
            if compute_single(forward_cutlist) > compute_single(backward_cutlist):
                return backward_cutlist
            else:
                return forward_cutlist

        elif count_backward > count_forward:
            return forward_cutlist

        else:
            return backward_cutlist





#测试
def test():
    domain_num=0
    cset=set()
    with open("../data_sets/Aleax","r") as f:
        for r in f:
            domain_num=domain_num+1
            if(domain_num>100):
                cset.add(r.strip().split(".")[0])
            if(domain_num>199):
                break
    for label in cset:
        sent = label
        print(sent)
        cuter = CutWords()
        print(cuter.max_forward_cut(sent))
        print(cuter.max_backward_cut(sent))
        wordlist=cuter.max_forward_cut(sent)
        vector = np.zeros(64)
        vi = 63
        for i in range(len(wordlist) - 1, -1, -1):
            vector[vi] = CutWords.order[wordlist[i]]
            vi = vi - 1
            if (vi < 0):
                break
        print(wordlist)
        print(vector)
    print("------------------------------")


    with open("../data_sets/wordlist.json","r") as f:
        worddga=json.load(f)


    for (k,v) in worddga.items():
        print(k)
        domain_num=0

        index=0
        for dga in v:
            index=index+1
            sent = dga.split(".")[0]
            print(sent)
            cuter = CutWords()
            # print(cuter.max_forward_cut(sent))
            # print(cuter.max_backward_cut(sent))
            wordlist=cuter.max_forward_cut(sent)
            vector=np.zeros(64)
            vi=63
            for i in range(len(wordlist) - 1, -1, -1):
                vector[vi]=CutWords.order[wordlist[i]]
                vi=vi-1
                if(vi<0):
                    break
            print(wordlist)
            print(vector)
            domain_num=domain_num+1
            if(domain_num>100):
                break



def lstm_getSingleFea(d:str):
    psl = PublicSuffixList()
    d = d[:d.rindex(psl.publicsuffix(d)) - 1].replace(".","")
    vector = np.zeros(64)
    if(len(d)==0):
        return vector
    cuter = CutWords()
    # wordlist = cuter.max_forward_cut(d)
    # wordlist = cuter.max_backward_cut(d)
    wordlist = cuter.max_biward_cut(d)

    vi = 63
    for i in range(len(wordlist) - 1, -1, -1):
        vector[vi] = CutWords.order[wordlist[i]]
        vi = vi - 1
        if (vi < 0):
            break
    # print(d)
    # print(vector)
    return vector

def lstm_getAllFea(domains):
    result=[lstm_getSingleFea(d) for d in domains]
    result=np.array(result)
    # print(result.shape)
    return result



if __name__=="__main__":
    domain="aliexpress"
    c=CutWords()
    print(c.max_forward_cut(domain))
    print(c.max_backward_cut(domain))

    print(c.max_biward_cut(domain))
