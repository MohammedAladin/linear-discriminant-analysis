import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

#using  LDA
iris = datasets.load_iris()
X = iris.data
y = iris.target

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.15)
# shuffle=True,


# pca = PCA(2)
# we need 2 principal components.
# converted_data_implemented =X
# converted_data_implemented = pca.fit_transform(X)
# converted_data_implemented.shape

# train_data = np.array(converted_data_implemented[:40])
# train_data = np.append(train_data , converted_data_implemented[51:91] , 0)
# train_data= np.append(train_data , converted_data_implemented[100:140] , 0)

# train_label = np.array(y[:40])
# train_label = np.append(train_label , y[51:91] , 0)
# train_label= np.append(train_label , y[100:140] , 0)

# test_data = np.array(converted_data_implemented[41:50])
# test_data = np.append(test_data , converted_data_implemented[91:100] , 0)
# test_data = np.append(test_data ,converted_data_implemented[141:150] , 0)


# test_labels = np.array(y[41:50])
# test_labels = np.append(test_labels , y[91:100] , 0)
# test_labels = np.append(test_labels, y[141:150] , 0)

#using  LDA
model = LinearDiscriminantAnalysis()
model.fit(train_data, train_labels)


new = test_data
predicted_labels = model.predict(new)

print("Predicted classes", predicted_labels)
print("Orignal classes",test_labels)

   
wrong_classifier = []
for i in range(len(test_labels)):
  if predicted_labels[i] != test_labels[i]:
    wrong_classifier.append(new[i])
 

wrong_predictions = len(wrong_classifier)
print("Wrong Classified Samples :", wrong_classifier)
accuracy = 100-((wrong_predictions/len(test_labels))*100)
print("test accuracy", accuracy)
#-------------------------------------------------------------------------------
#CREATE LDA PLOT
X_r2 = model.fit(X, y).transform(X)
target_names = iris.target_names

plt.figure()
colors = ['red', 'green', 'blue']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

#-------------------------------------------------------------------------------
#divide data

converted_data =X
# pca = PCA(2) # we need 2 principal components.
# converted_data = pca.fit_transform(X)
# converted_data.shape

#-----------------------------------------------
#getting w1
z1 = np.array(converted_data[:42])
z1 = np.append(z1, converted_data[50:92] , 0)
z1 = np.append(z1 , converted_data[100:142] , 0)
z1 = np.append(z1, [[1]*1]*126, axis=1)
b = [1]*126

for i in range (42,126):
  for j in range (5):
    z1[i][j] = -1.0* z1[i][j]

zT=z1.transpose()

w1 = [0]*5

b=np.array(b)
res1 = np.dot(zT,z1)
res2 = np.dot(zT,b)

matinv= np.linalg.inv(res1)
w1 = (np.dot(matinv,res2))
print("w1")
print(w1)

#--------------------------------------------------
#getting w2
z2 = np.array(converted_data[:42])
z2 = np.append(z2 , converted_data[100:142] , 0)
z2 = np.append(z2 , converted_data[50:92] , 0)
z2 = np.append(z2, [[1]*1]*126, axis=1)

for i in range (85):
  for j in range (5):
    z2[i][j] = -1.0* z2[i][j]

w2 = [0]*5
zT=z2.transpose()
res1 = np.dot(zT,z2)
res2 = np.dot(zT,b)

matinv= np.linalg.inv(res1)
w2 = (np.dot(matinv,res2))
print("w2")
print(w2)

# #--------------------------------------------------
#getting w3

z3 = np.array(converted_data[:42])
z3 = np.append(z3 , converted_data[50:92] , 0)
z3 = np.append(z3 , converted_data[100:142] , 0)
z3 = np.append(z3, [[1]*1]*126, axis=1)

for i in range (85):
  for j in range (5):
    z3[i][j] = -1.0* z3[i][j]

w3 = [0]*5
zT=z3.transpose()
res1 = np.dot(zT,z3)
res2 = np.dot(zT,b)

matinv= np.linalg.inv(res1)
w3 = (np.dot(matinv,res2))
print("w3")
print(w3)
#--------------------------------------------------
test_data = np.array(converted_data[42:50])
test_data = np.append(test_data , converted_data[92:100] , 0)
test_data = np.append(test_data ,converted_data[142:150] , 0)

labels = np.array(y[42:50])
labels = np.append(labels , y[92:100] , 0)
labels = np.append(labels, y[142:150] , 0)
print("Original Labels")
print(labels)

line1 = [0]*len(test_data)
line2 = [0]*len(test_data)
line3 = [0]*len(test_data)
labels_result = []

for i in range (len(test_data)):
  line1[i] = test_data[i][0]*w1[0] + test_data[i][1]*w1[1] +  test_data[i][2]*w1[2] + test_data[i][3]*w1[3] + w1[4]
  line2[i] = test_data[i][0]*w2[0] + test_data[i][1]*w2[1] +  test_data[i][2]*w2[2] + test_data[i][3]*w2[3] + w2[4]  
  line3[i] = test_data[i][0]*w3[0] + test_data[i][1]*w3[1] +  test_data[i][2]*w3[2] + test_data[i][3]*w3[3] + w3[4]
  if line1[i] > 0:
    if line2[i] < 0 and line3[i] < 0:
      labels_result.append(0)
    else:
      labels_result.append("Undetermined sample") 
        
  elif line2[i] > 0:
    if line1[i] < 0 and line3[i] < 0:
      labels_result.append(1)
    else:
      labels_result.append("Undetermined sample") 
   
  elif line3[i] > 0:
    if line1[i] < 0 and line2[i] < 0:
      labels_result.append(2)
    else:
      labels_result.append("Undetermined sample")

  elif line1[i] < 0 and line2[i] < 0 and line3[i] < 0:
     labels_result.append("newClass") 

wrong_manual_classifier = []
for i in range(len(test_labels)):
  if labels[i] != labels_result[i]:
    wrong_manual_classifier.append(test_data[i])

wrong_manual_classifier = np.array( wrong_manual_classifier)
wrong_predictions = len(wrong_manual_classifier)
print("Wrong Classified Samples : ")
print( wrong_manual_classifier)
accuracy = 100-((wrong_predictions/len(labels))*100)
print("Predicted Labels")
print(labels_result)
print("accuracy =", accuracy,"%")
