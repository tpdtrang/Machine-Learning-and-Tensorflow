from numpy import *
import operator
import numpy as np

def createDataSet(numofNeighbors):      
	neighbors = np.random.rand(numofNeighbors,2)      
	# assign A1 to the first neighbor, A2 to the second neighbor,...   
	names = ['']*numofNeighbors      
	for i in range(0,numofNeighbors):         
		names[i]= 'A'+str(i)     
	return neighbors, names

# For every point in our dataset:
def classify(me, neighbors, names, k):       
	numofNeighbors = neighbors.shape[0]   
	diffMat = tile (me, (numofNeighbors,1))-neighbors      
	# calculate the distance between me and the current neighbor     
	sqDiffMat = diffMat**2    
	sqDistances = sqDiffMat.sum(axis=1)    
	distancesMat = sqDistances**0.5   
	# sort the distances in increasing order   
	sortedDistIndicies = distancesMat.argsort()     
	# take k items with lowest distances to me
	classCount={}
	for i in range(k):     
		voteIname = names[sortedDistIndicies[i]]           
		classCount[voteIname] = classCount.get(voteIname,0) + 1      
	# find the majority class among these items      
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)    
	# return the majority class as our prediction for the class of me    
	return sortedClassCount

# Another version of classify function
def classify1(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1))-dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
	for w in sortedClassCount:
  		print(w[0])
