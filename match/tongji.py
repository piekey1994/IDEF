
hist=[0 for i in range(11)]

for i in range(8):
    begin=i*250000
    end=(i+1)*250000
    res_file_path='output/Twitter.v3%s-%s.score'%(begin,end)
    with open(res_file_path,'r',encoding='utf-8') as res_file:
        scores=[float(line.strip()) for line in res_file.readlines()]
        for score in scores:
            hist[int(score*10)]+=1

print(hist)
print([num/1000000 for num in hist])