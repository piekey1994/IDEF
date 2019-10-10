filename='test.oie'
with open(filename+'.orig','r',encoding='utf-8') as origFile,open(filename+'.sent','w',encoding='utf-8') as sentFile:
    sents=set()
    for line in origFile:
        sents.add(line.split('\t')[0]+'\n')
    sentFile.writelines(list(sents))