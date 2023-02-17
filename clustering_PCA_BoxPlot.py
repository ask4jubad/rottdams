# synthetic classification dataset
import numpy
from numpy import unique
from numpy import where
import re
import chart_studio.plotly as py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
from sklearn import datasets, decomposition
from sklearn.preprocessing import scale # for scaling the data
import sklearn.metrics as sm # for evaluating the model
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import sklearn.cluster as cluster
import time
import random
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import scipy
import vargha_delaney as VD


#cleaning up read values
def read_data(file):
    labels, labels_fail = [], []
    boxdata = [[],[],[]]
    counter, countF, countFf, countP = 0, 0, 0, 0
    data, datafail = [], []
    
    with open(file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:

        if line.find('Fail') != -1:
            countF += 1
            labels.append('Fail')
            labels_fail.append("Fail")
            line = re.sub(r'[A-Z: _a-z]+', '', line)
            line = re.sub(r'-', 'e-', line)
            line = [float(x) for x in line.split(",")]
            datafail.append(line)
            data.append(line)

        else:
            if line.find('False_fail') != -1:
                countFf += 1
                labels.append('False_fail')
                
            if line.find('Pass') != -1:
                countP += 1
                labels.append('Pass')

            line = re.sub(r'[A-Z: _a-z]+', '', line) #remove non numeric characters from result
            line = re.sub(r'-', 'e-', line) #remove non numeric characters from result
            line = [float(x) for x in line.split(",")] #convert each feature to float
            data.append(line)
            
        counter += 1
        
        if (counter % 400) == 0:
           boxdata[0].append(countF)
           boxdata[1].append(countFf)
           boxdata[2].append(countP)
           
           countF, countFf, countP = 0, 0, 0

    return data, boxdata, labels
    
dataGA, boxGA, labelsGA = read_data("result_GA.txt")
dataRS, boxRS, labelsRS = read_data("result_RS.txt")
dataES, boxES, labelsES = read_data("result_ES.txt")
dataSA, boxSA, labelsSA = read_data("result_SA.txt")


ticks = ["Fail", "False_fail", "Pass"]

# # plt.subplot(2,2,1)


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()

for i in range(1, 4):
    plt.subplot(2,2,i)
    GA_plot = plt.boxplot(boxGA[i-1], positions=numpy.array(numpy.arange(len([boxGA[i-1]])))*2.0-1.5,widths=0.25, showmeans=True)
    # Save the histogram
    # plt.xticks(ticks=numpy.array(numpy.arange(len([boxGA[i-1]])))*2.0-1 ,labels= ["RS"])
    RS_plot = plt.boxplot(boxRS[i-1], positions=numpy.array(numpy.arange(len([boxRS[i-1]])))*2.0-1,widths=0.25, showmeans=True)
    # plt.xticks(ticks=numpy.array(numpy.arange(len([boxGA[i-1]])))*2.0-0.5 ,labels= ["ES"])
    ES_plot = plt.boxplot(boxES[i-1], positions=numpy.array(numpy.arange(len([boxES[i-1]])))*2.0-0.5,widths=0.25, showmeans=True)
    # plt.xticks(ticks=numpy.array(numpy.arange(len([boxGA[i-1]])))*2.0,labels= ["SA"])
    SA_plot = plt.boxplot(boxSA[i-1], positions=numpy.array(numpy.arange(len([boxSA[i-1]])))*2.0,widths=0.25, showmeans=True)
    # plt.xticks(ticks=numpy.array(numpy.arange(len([boxGA[i-1]])))*2.0,labels= ["SA"])
    # print(numpy.median(boxGA[i-1]), numpy.median(boxRS[i-1]), numpy.median(boxES[i-1]), numpy.median(boxSA[i-1]))
    # print(numpy.median(boxGA[i-1]), numpy.median(boxRS[i-1]), numpy.median(boxES[i-1]), numpy.median(boxSA[i-1]))
    plt.grid(linestyle='-', linewidth=0.5, axis='y')
    
    # # setting colors for each groups
    # define_box_properties(GA_plot, '#D7191C', 'GA')
    # define_box_properties(RS_plot, '#2C7BB6', 'RS')
    # define_box_properties(ES_plot, '#223344', 'ES')
    # define_box_properties(SA_plot, '#00ff00', 'SA')
    
    # setting colors for each groups
    define_box_properties(GA_plot, '#c60990', 'GA')
    define_box_properties(RS_plot, '#244b5c', 'RS')
    define_box_properties(ES_plot, '#7a67ee', 'ES')
    define_box_properties(SA_plot, '#30ba8f', 'SA')
        
    plt.xticks(ticks=[])
    # plt.xticks(ticks=None ,labels= ["GA", "RS", "ES", "SA"])
    plt.xlabel(ticks[i-1])

    # # set the title
    plt.title('Grouped boxplot for {} evaluations for GA, RS, ES and SA'.format(ticks[i-1]))
    plt.ylabel("No of evaluations")
    

plt.subplot(2,2,4)
for i in range(1):
        # plt.plot([str(j+1) for j in range(20)], boxGA[i], label = "GA_" + ticks[i], marker='o', linewidth=1, c='#D7191C')
        # plt.plot([str(j+1) for j in range(20)], boxRS[i], label = "RS_" + ticks[i], marker='x', linewidth=1, c='#2C7BB6')    
        # plt.plot([str(j+1) for j in range(20)], boxES[i], label = "ES_" + ticks[i], marker='*', linewidth=1, c='#223344')
        # plt.plot([str(j+1) for j in range(20)], boxSA[i], label = "SA_" + ticks[i], marker='h', linewidth=1, c='#00ff00')
        plt.plot([str(j+1) for j in range(20)], boxGA[i], label = "GA_" + ticks[i], marker='o', linewidth=1, c='#c60990')
        plt.plot([str(j+1) for j in range(20)], boxRS[i], label = "RS_" + ticks[i], marker='x', linewidth=1, c='#244b5c')    
        plt.plot([str(j+1) for j in range(20)], boxES[i], label = "ES_" + ticks[i], marker='*', linewidth=1, c='#7a67ee')
        plt.plot([str(j+1) for j in range(20)], boxSA[i], label = "SA_" + ticks[i], marker='h', linewidth=1, c='#30ba8f')

plt.legend(loc='upper right')
plt.grid(linestyle=':', linewidth=1, axis='both')
plt.title('Distribution of number of fail evaluations per simulation run')
plt.xlabel("Simulation run")
plt.ylabel("No of evaluations")
plt.show()



def extractfaileddata(data, labels):
    buffer = []
    for i in range(len(data)):
        if labels[i] == "Fail":
            buffer.append(data[i])
    
    return buffer
            
dataGAfail = (extractfaileddata(dataGA, labelsGA))
dataRSfail = (extractfaileddata(dataRS, labelsRS))
dataESfail = (extractfaileddata(dataES, labelsES))
dataSAfail = (extractfaileddata(dataSA, labelsSA))

def clusterOutput(data):

    # load the data
    X = numpy.array(data, dtype=object)
    # print(X)

    # perform PCA to reduce the dimensionality of the data
    pca = PCA()
    X_reduced = pca.fit_transform(X)
    # print(X_reduced)

    # determine the number of components to keep based on the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    print(explained_variance)
    
    # plt.plot(explained_variance)
    # plt.show()
    
    num_components = numpy.argmin(numpy.cumsum(explained_variance) < 0.95) + 1

    # reduce the data to the number of components determined above
    X_reduced = X_reduced[:, :num_components]
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    # plt.show()


    # perform clustering on the reduced data
    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X_reduced)
    
    # plot the reduced data
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters)
    plt.show()
    
    return num_components, clusters
    
# plt.subplot(2,2,1)
print(clusterOutput(dataGAfail)[0], len(unique(clusterOutput(dataGAfail)[1])))
# print(unique(clusterOutput(dataGAfail)[1]))
# plt.subplot(2,2,2)
# print(clusterOutput([dataRS[0]]))
# print(unique(clusterOutput(dataRSfail)[1]))
print(clusterOutput(dataESfail)[0], len(unique(clusterOutput(dataESfail)[1])))
print(clusterOutput(dataSAfail)[0], len(unique(clusterOutput(dataSAfail)[1])))
print(clusterOutput(dataRSfail)[0], len(unique(clusterOutput(dataRSfail)[1])))
# # plt.subplot(2,2,3)
# print(clusterOutput([dataES[0]]))
# # plt.subplot(2,2,4)
# print(clusterOutput([dataSA[0]]))

# # plt.show()

# X = numpy.array(dataGA, dtype=object)
# dbscan = DBSCAN()
# clusters = dbscan.fit_predict(X)

# print(unique(clusters))

print(scipy.stats.mannwhitneyu(boxGA[0], boxRS[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxGA[0], boxES[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxGA[0], boxSA[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxES[0], boxRS[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxES[0], boxGA[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxES[0], boxSA[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxSA[0], boxRS[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxSA[0], boxGA[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxSA[0], boxES[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxRS[0], boxGA[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxRS[0], boxES[0], method='auto'))
print(scipy.stats.mannwhitneyu(boxRS[0], boxSA[0], method='auto'))


print("\n==========================================================\n")
print(VD.VD_A(boxGA[0], boxRS[0]))
print(VD.VD_A(boxGA[0], boxES[0]))
print(VD.VD_A(boxGA[0], boxSA[0]))
print(VD.VD_A(boxES[0], boxRS[0]))
print(VD.VD_A(boxES[0], boxGA[0]))
print(VD.VD_A(boxES[0], boxSA[0]))
print(VD.VD_A(boxSA[0], boxRS[0]))
print(VD.VD_A(boxSA[0], boxES[0]))
print(VD.VD_A(boxSA[0], boxRS[0]))
print(VD.VD_A(boxRS[0], boxGA[0]))
print(VD.VD_A(boxRS[0], boxES[0]))
print(VD.VD_A(boxRS[0], boxSA[0]))
