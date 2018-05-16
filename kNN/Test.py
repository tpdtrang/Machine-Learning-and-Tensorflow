from numpy import *
import matplotlib.pyplot as plt
import kNN
# create 5 my neighbors
neighbors,names = kNN.createDataSet(5)
# find two nearest neighbors to me
result = kNN.classify([0,0], neighbors, names, 2)
# x and y save positons of my neighbors
x = [0]* neighbors.shape[0]
y = [0]* neighbors.shape[0]
for i in range(0, neighbors.shape[0]):   
	x[i] = neighbors [i][0]      
	y[i] = neighbors [i][1]
# display my neighbors with blue color
plt.plot(x,y,'bo')
plt.axis([-0.2, 1.2, -0.2, 1.2])
# assign names to neighbors
for i, name in enumerate(names):     
plt.annotate(name,(x[i],y[i]),(x[i]-0.08,y[i]+0.01))
# diplay me with red color
plt.plot([0],[0],'ro')
# display two nearest neighbors with  messages and yellow color
for i, name in enumerate(names):    
	for r in result:           
		if name is r[0]:               
			plt.plot([x[i]],[y[i]],'yo')                  
			plt.annotate('I am here',(x[i],y[i]),(x[i]+0.01,y[i]-0.05))
plt.show()
