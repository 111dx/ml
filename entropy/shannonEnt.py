import numpy as np
#return :数据集(list),类别名字(list)
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels
#计算香农熵 params=[类别值(iter)] ,各个值必须是int
def calcShannonEnt(cValue):
    #计算各类别频数
    cCount =np.bincount(cValue)
    cCount =cCount[cCount!=0]       #把为零的去掉
    #计算熵
    ent =0
    sizeData =len(cValue)
    prob =cCount/sizeData
    ent =-sum(prob*np.log2(prob))
    return ent
#切分数据集 params=[数据集(arr),特征的索引(int),特征值
def splitDataSet(arrData,axis,value):
    '''数据集(arr)，特征(int)，值'''
    #用布尔计算方式进行划分
    index =arrData[:,axis] ==value
    dataSplit =arrData[index]
    return  np.delete(dataSplit,axis,axis=1)
#选择最好特征 params =[数据集(arr),类别值(iter)] ,各个值必须是int  return =特征的索引(int)
def chooseBestFeatureToSplit(dataSet,cValues):
    '''数据集(arr)'''
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(cValues)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        #特征所有取值
        arrFeature =set(dataSet[:,i])
        #计算所有划分出子集的熵
        newEntropy = 0.0
        for value in arrFeature:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet[:,-1])
        #计算信息增益
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer
#创建决策树 params =[数据集(arr),类别名字(iter)]
def createTree(arrData,labels):
    cValues =arrData[:,-1]  #类别值
    #判断停止
    if sum(cValues==cValues[0]) ==cValues.size:     #如果全部类值相等
        return cValues[0]
    if (arrData[0].size) ==1 or labels==0:          #特征用完，则返回出现次数最多的类值
        return cValues[np.argmax(np.bincount(cValues))]
    #选择最好特征
    featureBest =chooseBestFeatureToSplit(arrData,cValues)      #特征索引
    labelBest =labels[featureBest]                      #特征名字
    #用字典构建决策树
    tree ={labelBest:{}}
    arrFeature = set(arrData[:,featureBest])            #特征取值
    del labels[featureBest]                             #删除特征名字
    subLabels =labels[:]                                #复制
    for i in arrFeature:
        tree[labelBest][i] =createTree(splitDataSet(arrData,featureBest,i),subLabels)
    return tree


if __name__ =='__main__':
    #得到数据
    data,labels =createDataSet()
    #将分类变量替换成数值
    data=np.array(data)
    dictMap ={'yes':1,'no':0}
    data[:, -1] = list(map(lambda x: dictMap[x], data[:, -1]))
    #建立决策树
    a =createTree(data.astype('i1'),labels)

    print(a)

