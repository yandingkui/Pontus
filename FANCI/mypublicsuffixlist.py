import os
current_path = os.path.dirname(__file__)

def clean_domain_list(domain_list: list, dga=False):
    """
    Cleans a given domain list from invalid domains and cleans each single domain in the list.
    :param domain_list:
    :param dga:
    :return:
    """

    domain_list = [d.strip().lower() for d in domain_list]
    domain_list = list(filter(None, domain_list))

    if dga:
        # some ramnit domains ending with the pattern: [u'.bid', u'.eu']
        to_remove = []
        for d in domain_list:
            if '[' in d:

                to_remove.append(d)
                res = set()
                bracket_split = d.split('[')
                tlds = bracket_split[1].split(',')
                for tld in tlds:
                    tld = tld.strip()
                    tld = tld.split("'")[1].replace('.', '')
                    res_d = bracket_split[0] + tld
                    res.add(res_d)
                    domain_list.append(res_d)

        domain_list = [d for d in domain_list if d not in to_remove]

    return domain_list




class myPublicSuffixes:
    """
    Represents the official public suffixes list maintained by Mozilla  https://publicsuffix.org/list/
    """
    def __init__(self, file="../source/public_suffix.txt"):
        #print(os.path.abspath(file))
        with open(file, encoding='utf-8',mode="r") as f:
            self.data = f.readlines()

        self.data = clean_domain_list(self.data)
        self.data = ['.' + s for s in self.data if not (s.startswith('/') or s.startswith('*'))]
        self.data = clean_domain_list(self.data)

    def get_valid_tlds(self):
        tlds=dict()
        for s in self.data:
            s.strip()
            if len(s.split("."))==2:
                tlds[s]=True
        return tlds
        #return [s for s in self.data if len(s.split('.')) == 2]

    def get_valid_public_suffixes(self):
        return self.data

if __name__=="__main__":
    # pl=myPublicSuffixes()
    # print(pl.get_valid_tlds())

    print(os.path.abspath("../referenceFiles/public_suffix.txt"))
    with open("../referenceFiles/public_suffix.txt",mode="r") as f:
        print(len(f.readlines()))