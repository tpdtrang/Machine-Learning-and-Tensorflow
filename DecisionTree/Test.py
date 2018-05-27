import decisionTree

# create a dataset and labels
myDat,labels= decisionTree.createDataSet()
print(myDat)
# calculate the entropy of the dataset
entropy = decisionTree.calcEntropy(myDat)
print(entropy)
# calculate information gain of A feature
gainA = decisionTree.infoGain(myDat,0)
print(gainA)
# calculate information gain of B feature
gainB = decisionTree.infoGain(myDat,1)
print(gainB)
# split the dataset based on A feature and its 1 value
splitA = decisionTree.splitDataSet(myDat,0,1)
print(splitA)
# choose the best feature
best = decisionTree.chooseBestFeature(myDat)
print(best)
# create my decision tree
tree = decisionTree.createTree(myDat,labels,0)
print(tree)
