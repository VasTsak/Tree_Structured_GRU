import re
import os
import pandas as pd
import numpy as np
import matplotlib as plt

# Set working directory
os.chdir("...")

## Read the files 

#Time 
binary_gru = open("binary_gru.txt", "r")
binary_lstm = open("binary_lstm.txt", "r")
fine_grained_gru = open("fine_grained_gru.txt", "r")
fine_grained_lstm = open("fine_grained_lstm.txt", "r")
time_binary_gru = re.findall(r'epochis (\d+\.\d+)',binary_gru.read())
time_binary_lstm =  re.findall(r'epochis (\d+\.\d+)',binary_lstm.read())
time_fine_grained_gru = re.findall(r'epochis (\d+\.\d+)',fine_grained_gru.read())
time_fine_grained_lstm = re.findall(r'epochis (\d+\.\d+)',fine_grained_lstm.read())
#time_binary_lstm = loss_binary_lstm[:-1]
time = pd.DataFrame(columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,25,1)])

for i in xrange(25):
    time.iloc[i,0] = time_binary_gru[i]
    time.iloc[i,1] = time_binary_lstm[i]
    time.iloc[i,2] = time_fine_grained_gru[i]
    time.iloc[i,3] = time_fine_grained_lstm[i]

# Loss scores
binary_gru = open("binary_gru.txt", "r")
binary_lstm = open("binary_lstm.txt", "r")
fine_grained_gru = open("fine_grained_gru.txt", "r")
fine_grained_lstm = open("fine_grained_lstm.txt", "r")
type(loss_binary_gru)
loss_binary_gru = re.findall(r'avg loss (\d+\.\d+)',binary_gru.read())
loss_binary_lstm =  re.findall(r'avg loss (\d+\.\d+)',binary_lstm.read())
loss_fine_grained_gru = re.findall(r'avg loss (\d+\.\d+)',fine_grained_gru.read())
loss_fine_grained_lstm = re.findall(r'avg loss (\d+\.\d+)',fine_grained_lstm.read())
loss_binary_lstm = loss_binary_lstm[:-1]
loss = pd.DataFrame(columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,25,1)])

for i in xrange(25):
    loss.iloc[i,0] = loss_binary_gru[i]
    loss.iloc[i,1] = loss_binary_lstm[i]
    loss.iloc[i,2] = loss_fine_grained_gru[i]
    loss.iloc[i,3] = loss_fine_grained_lstm[i]

# Validation score
binary_gru = open("binary_gru.txt", "r")
binary_lstm = open("binary_lstm.txt", "r")
fine_grained_gru = open("fine_grained_gru.txt", "r")
fine_grained_lstm = open("fine_grained_lstm.txt", "r")

dev_binary_gru = re.findall(r'dev-scoer (\d+\.\d+)',binary_gru.read())
dev_binary_lstm = re.findall(r'dev-scoer (\d+\.\d+)',binary_lstm.read())
dev_fine_grained_gru = re.findall(r'dev-scoer (\d+\.\d+)',fine_grained_gru.read())
dev_fine_grained_lstm  = re.findall(r'dev-scoer (\d+\.\d+)',fine_grained_lstm.read())

dev = pd.DataFrame(columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,25,1)])

for i in xrange(25):
    dev.iloc[i,0] = dev_binary_gru[i]
    dev.iloc[i,1] = dev_binary_lstm[i]
    dev.iloc[i,2] = dev_fine_grained_gru[i]
    dev.iloc[i,3] = dev_fine_grained_lstm[i]


#test score

binary_gru = open("binary_gru.txt", "r")
binary_lstm = open("binary_lstm.txt", "r")
fine_grained_gru = open("fine_grained_gru.txt", "r")
fine_grained_lstm = open("fine_grained_lstm.txt", "r")

test_binary_gru = re.findall(r'(\d+\.\d+) test_score',binary_gru.read())
test_binary_lstm = re.findall(r'(\d+\.\d+) test_score',binary_lstm.read())
test_fine_grained_gru = re.findall(r'(\d+\.\d+) test_score',fine_grained_gru.read())
test_fine_grained_lstm  = re.findall(r'(\d+\.\d+) test_score',fine_grained_lstm.read())

test = pd.DataFrame(columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,5,1)])
for i in xrange(5):
    test.iloc[i,0] = test_binary_gru[i]
    test.iloc[i,1] = test_binary_lstm[i]
    test.iloc[i,2] = test_fine_grained_gru[i]
    test.iloc[i,3] = test_fine_grained_lstm[i]

# Make the data frame values floats
loss = loss.astype(float)
dev = dev.astype(float)
test = test.astype(float)
time = time.astype(float)

# create the averages
avg_time = pd.DataFrame(np.zeros((5, 4)), columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,5,1)])
avg_loss = pd.DataFrame(np.zeros((5, 4)), columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,5,1)])
dev_scores = pd.DataFrame(np.zeros((5, 4)),columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [np.arange(0,5,1)])
test_scores = pd.DataFrame(columns=['binary_gru','binary_lstm','fine_grained_gru','fine_grained_lstm'], index = [1])
# 4 columns
for i in xrange(4):
    # 5 iterations
    for j in xrange(5):
        # step of 5
        for k in range(0,21,5):
            avg_time.iloc[j,i] +=0.2* time.iloc[j+k,i] 

# 4 columns
for i in xrange(4):
    # 5 iterations
    for j in xrange(5):
        # step of 5
        for k in range(0,21,5):
            avg_loss.iloc[j,i] +=0.2* loss.iloc[j+k,i] 
            
for i in xrange(4):
    for j in xrange(5):
        for k in range(0,21,5):
            dev_scores.iloc[j,i] += 0.2 * dev.iloc[j+k,i] 
        
        
for i in xrange(4):
    test_scores.iloc[0,i] = test[[i]].sum().values/5

#plots for avg_loss

plt.figure(1)
plt.subplot(211)
plt.plot(avg_loss[[0]] ,'r-' ,label = "Tree-GRU")
plt.plot(avg_loss[[1]], 'g-', label = "Tree-LSTM")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(212)
plt.plot(avg_loss[[2]] ,'r-' ,label = "Tree-GRU")
plt.plot(avg_loss[[3]], 'g-', label = "Tree-LSTM")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# plots for average dev scorre

plt.figure(1)
plt.subplot(211)
plt.plot(dev_scores[[0]] ,'r-' ,label = "Tree-GRU")
plt.plot(dev_scores[[1]], 'g-', label = "Tree-LSTM")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(212)
plt.plot(dev_scores[[2]] ,'r-' ,label = "Tree-GRU")
plt.plot(dev_scores[[3]], 'g-', label = "Tree-LSTM")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.figure(1)
plt.subplot(211)
plt.plot(avg_time[[0]] ,'r-' ,label = "Tree-GRU")
plt.plot(avg_time[[1]], 'g-', label = "Tree-LSTM")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(212)
plt.plot(avg_time[[2]] ,'r-' ,label = "Tree-GRU")
plt.plot(avg_time[[3]], 'g-', label = "Tree-LSTM")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
