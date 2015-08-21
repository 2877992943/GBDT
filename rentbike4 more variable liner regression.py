import random
import os
import sys
import math
 


inpath = "D://python2.7.6//MachineLearning//gbdt//datatxt"
outfile1 = "D://python2.7.6//MachineLearning//gbdt//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//gbdt//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//gbdt//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//gbdt//4.txt"
outfile5 = "D://python2.7.6//MachineLearning//gbdt//5.txt"
     

eps=0.00001  
######################

def loadData():
    global dataDic;dataDic={};global featList;featList=[]
    for filename in os.listdir(inpath):
        
        content=open(inpath+'/'+filename,'r')#.strip()
        feat=content.readline().strip('\n')
        if feat not in dataDic:
            dataDic[feat]=[]
            featList.append(feat)

        data=[]
        line=content.readline()
        while line:
            if line.strip('\n')<1:data.append(-10)
            else:data.append(line.strip('\n'))
            line=content.readline()
        dataDic[feat]=data
        #print len(data),data[-1],data[0]
        global numRecord;numRecord=len(data)
        #if -10 in data:print data.index(-10)
                            

    #####################################datetime
    dataDic['year']=[];dataDic['month']=[];dataDic['day']=[];dataDic['time']=[]
    for date in dataDic['datetime']:
        date=date.replace('/',' ').split(' ')
        dataDic['year'].append(int(date[0]))
        dataDic['month'].append(int(date[1]))
        dataDic['day'].append(int(date[2]))
        pos=date[3].index(':')
        dataDic['time'].append(int(date[3][:pos]))
    featList.append('year')
    featList.append('month')
    featList.append('day')
    featList.append('time')
    featList.remove('datetime')
    featList.remove('casual')
    featList.remove('count')
    featList.remove('registered')
    global yList;yList=['casual','registered','count']
    #print featList
    
    ###############################
    global recordList;recordList=[]
    for i in range(numRecord):
        record=[{},{},'predict',{}]
        for feat in featList:
            record[0][feat]=float(dataDic[feat][i])

        for y in yList:
            record[1][y]=float(dataDic[y][i])
            record[3][y]=float(dataDic[y][i])
        recordList.append(record)
    
    ################add xi^2
    for fea in featList:
        for fea1 in featList:
            newf=fea+'-'+fea1
            for re in recordList:
                re[0][newf]=re[0][fea]*re[0][fea1]
                
    for fea in recordList[0][0].keys():
        if fea not in featList:
            featList.append(fea)
    ###############add bias to feat xi
    for re in recordList:
        re[0]['b']=1.0
    #####################
    outPutfile=open(outfile1,'w')
    for re in recordList:
        outPutfile.write(str(re[0]));
        outPutfile.write('\n')
        outPutfile.write(str(re[1]));
        outPutfile.write('\n')
    outPutfile.close()
        

            


############################main
loadData()
#calcIxy()

