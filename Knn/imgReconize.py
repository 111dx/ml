from numpy import *
from os import listdir

def loadData(dirname):
    listData =listdir(dirname)
    #把32*32文本文件读为1*1024
    def file2arr(filename):
        with open(filename) as f:
            vec =f.read()
        return list(vec.replace('\n',''))
    sizeData =len(listData)         #文件的数量
    cLabel=zeros(sizeData,dtype='int16')          #文件类别
    arrTrain =zeros((sizeData,1024),dtype='int16')  #训练样本数组
    for i,j in enumerate(listData):
        cLabel[i] =int(j[0]) #每个文件对应的类别
        arrTrain[i,:] =file2arr(dirname + '\\' +j)
    return cLabel,arrTrain

if __name__ =='__main__':
    from knn import classify0
    fTrain = r'..\data\Ch02\digits\trainingDigits'
    fTest = r'..\data\Ch02\digits\testDigits'
    cLabel,arrTrain =loadData(fTrain)
    cLabelTest,arrTest =loadData(fTest)
    err=0
    for j,i in enumerate(arrTest):
        label =classify0(i,arrTrain,cLabel,3)
        if cLabelTest[j] !=label:err+=1
    print('错误率：',err/len(cLabelTest))

    #sklearn库knn对比
    from sklearn.neighbors import KNeighborsClassifier as knn
    model =knn(n_neighbors=3,n_jobs=4,algorithm='auto')
    model.fit(arrTrain,cLabel)
    cLabelPredict =model.predict(arrTest)
    print('错误率',sum(cLabelPredict!=cLabelTest)/len(cLabelTest))
