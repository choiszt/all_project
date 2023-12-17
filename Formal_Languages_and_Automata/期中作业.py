import re
with open('nfa1.txt', 'r') as f:
    list=f.readlines()
    f.close()
startlist=[]
endlist=[]
dict1,dict2={},{}
num_status=len(list[0].split(' ')) #状态数

#处理字符串，匹配左右括号
for num in range(len(list)):
    leftsign=[]
    rightsign=[]
    if list[num].startswith('#'):
        startlist.append(list[num][1:3])
    if list[num].startswith('*'):
        endlist.append(list[num][1:3])
    for i in (re.finditer(r'{',list[num])):
        # print(i.span()[0])
        leftsign.append(i.span()[0])
        dict1[f'q{num}']=leftsign
    for i in (re.finditer(r'}',list[num])):
        # print(i.span()[0])
        rightsign.append(i.span()[0])
        dict2[f'q{num}']=rightsign

result={}#状态转移表
# print(dict1)
# print(dict2)

#建立状态映射
for itm in dict1.items():
    for num in range(num_status):
        if dict2[itm[0]][num]-dict1[itm[0]][num]==1:
            # print(dict1[itm[0]],dict2[itm[0]])
            # print(dict2[itm[0]][num],dict1[itm[0]][num])
            result[(itm[0],list[0].split(' ')[num].rstrip())]=None#第一行状态的哪一项
        else:
            substart=dict1[itm[0]][num]
            subend=dict2[itm[0]][num]
            element=[] #每一项直接连接的集合
            for ele in (list[int(itm[0][1:])][substart+1:subend].split(',')):
                element.append(ele)
                result[(itm[0],list[0].split(' ')[num].rstrip())]=element #将状态及其相连状态存入字典
# print(result)

def Eclosure(q,closure):
    if result[(q,'epsilon')] is not None:
        for ele in result[(q,'epsilon')]:
            if ele not in closure:
                # print(ele)
                closure.append(ele)
                Eclosure(ele,closure)
    return closure

def findclosure(q,input):
    nfaclosure=[]
    closure=[]
    backclosure=[]
    if q not in Eclosure(q,closure=closure):
        closure.append(q)
    #print(f"closure of {q}=",closure)
    for ele in closure:#[q1,q2,q3]
        if result[(ele,input)] is not None:
            for i in result[(ele,input)]:
                if i not in nfaclosure:
                    nfaclosure.append(i)
                    if i not in Eclosure(i,closure=backclosure):
                        backclosure.append(i)
                    for item in backclosure:
                        if item not in nfaclosure:
                            nfaclosure.append(item)
    return nfaclosure


dfadict={}
a=[]
for i in result.keys():
    para1,para2=i
    if para2=='epsilon' :
        if para1 not in findclosure(para1, para2):
            a= findclosure(para1,para2)
            a.append(para1)
            dfadict[(para1,para2)]=a
    else:
        dfadict[(para1,para2)]=findclosure(para1,para2)

startclosure=[]
dfaendlist=endlist
for ele in startlist:
    new=[]
    new=Eclosure(ele,startclosure)
    new.append(ele)
    for end in endlist:
        if end in new:
            if ele not in dfaendlist:
                dfaendlist.append(ele)

print(dfadict)

#convert to DFA
s= {1}
for ele in dfadict.keys():
    s.clear()
    for i in dfadict[ele]:
        s.add(i)
    # print(ele,s)
    set=s.copy()
    dfadict[ele]=set
# print(list[0].split(' ')[:-1])


status=startlist
new=[]
for i in startlist:
    new.append({i})
for i in startlist:
    for num in list[0].split(' ')[:-1]:#除去空移动的集合进行转化
        # print(dfadict[i,num])
        if dfadict[i,num] not in new:
            new.append(dfadict[i,num])

for i in range(2**len(list[0])):
    tempset={1}
    tempset.clear()
    for ele in new:
        # print('ele=',ele)
        for num in list[0].split(' ')[:-1]:
            # print(num)
            for sta in ele:#[q1,q2]
                for i in dfadict[sta,num]:#[q1,1]
                    tempset.add(i)
            # print('temp=',tempset)
            rand=tempset.copy()
            if rand not in new and len(rand)!=0:
                new.append(rand)
            #     print('temp=',tempset)
            # print(tempset)
            tempset.clear()

# print(new)
finaldict={}
for ele in new:
    for num in list[0].split(' ')[:-1]:
        finallist=[]
        for sta in ele:
            for i in dfadict[sta,num]:
                if i not in finallist:
                    finallist.append(i)
        finaldict[tuple(ele),num]=finallist
# print(finaldict)
with open('dfa1.txt', 'w+') as f:
    for num in list[0].split(' ')[:-1]:
        f.write(num)
        f.write(" ")
    for i in new:
        f.write('\n')
        for end in dfaendlist:
            if end in i:
                f.write("*")
                break
        for start in startlist:
            if(f"{{'{start}'}}")==str(i):
                f.write("#")
                break
        f.write(str(i))
        f.write(' ')

        for num in list[0].split(' ')[:-1]:
            # print((str(tuple(i)),num))
            if(len(finaldict[(tuple(i)),num])!=0):
                f.write(str(finaldict[(tuple(i)),num]))





