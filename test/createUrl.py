def createUrl(path):
    result=[]
    with open(path,"r") as f:
        for r in f:
            r=r.strip()
            result.append("https://gdc.xenahubs.net/download/TCGA-"+r+"/Xena_Matrices/TCGA-"+r+".htseq_counts.tsv.gz")
            result.append("https://gdc.xenahubs.net/download/TCGA-"+r+"/Xena_Matrices/TCGA-"+r+".htseq_fpkm.tsv.gz")
            result.append("https://gdc.xenahubs.net/download/TCGA-"+r+"/Xena_Matrices/TCGA-"+r+".htseq_fpkm-uq.tsv.gz")
    with open(path+".ok","w") as f:
        f.write("\n".join(result))

if __name__=="__main__":
    createUrl("/Users/yandingkui/Desktop/TCGA.txt");