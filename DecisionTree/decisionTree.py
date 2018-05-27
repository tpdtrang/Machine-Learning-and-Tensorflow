from math import log
import operator

def createDataSet():
  dataSet = [[1, 1, 'yes'],
		[1, 1, 'yes'],
		[0, 1, 'no'],
		[1, 0, 'no'],
		[1, 0, 'no']]
  labels = ['A','B']
  return dataSet, labels

# calculate entropy of dataset
def calcEntropy(dataSet):
	# number of data instances or number of rows of our dataSet
	numIntances = len(dataSet)
  	# create a dictionary whose keys are the values in the final column
	keyCounts = {}
  	# for each instance or row 
	for instance in dataSet:
		# return value in the final column
		currentKey = instance[-1]
		# If a key was not encountered previously, one is created
		if currentKey not in keyCounts.keys(): keyCounts[currentKey] = 0
		# For each key, keep track of how many times this key occurs
		keyCounts[currentKey] += 1
	# our entropy
	entropy = 0.0
	# for each key
	for key in keyCounts:
		# calculate the probability of this key
		prob = float(keyCounts[key])/numIntances 
		# calculate entropy
		entropy -= prob * log(prob,2)
	return entropy

# split the dataset based on a specific feature and its value
def splitDataSet(dataSet, feature, value):
	subDataSet = []
	for row in dataSet:
		if row[feature] == value:
			reducedRow = row[:feature]
			reducedRow.extend(row[feature+1:])
			subDataSet.append(reducedRow)
	return subDataSet
 
def infoGain(dataSet, feature):
	# calculate entropy of dataset
	baseEntropy = calcEntropy(dataSet)
	# entropy of subdataset
	newEntropy = 0.0
	featList = [example[feature] for example in dataSet]
	# Get list of unique values, ex: [0,0,1,1,1] -> {0,1}
	uniqueVals = set(featList)
	# iterate over all the unique values of this feature and split the data # for each feature
	for value in uniqueVals:
		# split dataset based on feature and value
		subDataSet = splitDataSet(dataSet, feature, value)	
		# The new entropy is calculated and summed up for all the unique	
		# values of that feature
		prob = len(subDataSet)/float(len(dataSet))
		newEntropy += prob * calcEntropy(subDataSet)
		# the information gain of feature is the reduction in entropy
	gain = baseEntropy - newEntropy
	return gain

def chooseBestFeature(dataSet):
	numFeatures = len(dataSet[0]) - 1
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		  gain = infoGain(dataSet,i)
		  if (gain > bestInfoGain):
			  bestInfoGain = gain
			  bestFeature = i
	return bestFeature
 
# find the majority result class ('yes' or 'no') that occurs with the greatest frequency.
def majorityClass(classList):
  classCount={}
  for vote in classList:
	    if vote not in classCount.keys(): classCount[vote] = 0
	    classCount[vote] += 1
  sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

# create my decision tree
def createTree(dataSet,labels,level):
  classList = [example[-1] for example in dataSet]
  # Stop when all result classes are equal
  if classList.count(classList[0]) == len(classList):
	    return classList[0]
  # When no more features, return majority class
  if len(dataSet[0]) == 1:
	    return majorityClass (classList)
  bestFeat = chooseBestFeature(dataSet)	
  bestFeatLabel = labels[bestFeat]
  if level == 0:
	    myTree = "Root =>   " + bestFeatLabel + '\n\n'
  else:
	    myTree = bestFeatLabel + '\n\n' 	
  del(labels[bestFeat])
  featValues = [example[bestFeat] for example in dataSet]
  # Get list of unique values, ex: [0,0,1,1,1] -> {0,1}
  uniqueVals = set(featValues)
  # levels of tree
  level = level + 1
  myTree += "Level " + str(level) + " =>   " 
  # iterate over all the unique values from our chosen feature and recursively 
  # call createTree() for each split of the dataset
  for value in uniqueVals:
	    subLabels = labels[:]
	    myTree += 'Branch ' + str(value)+ ' of ' + bestFeatLabel + ' : '	
	    myTree += createTree(splitDataSet(dataSet, bestFeat, value),subLabels,level) + ' '				
  return myTree


