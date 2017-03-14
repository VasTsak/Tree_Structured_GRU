import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
 
os.chdir(".../results")
 
# Note that the name of the variables indicate the classifiaction method and the used model.
# Read the csv from resutls and remove the first column - read.csv bug
for i in range(1,6):
    locals()["binary_gru_"+str(i)] = pd.read_csv("binary_gru_"+str(i)+".csv").iloc[:,range(1,6)]
    locals()["binary_lstm_"+str(i)] = pd.read_csv("binary_lstm_"+str(i)+".csv").iloc[:,range(1,6)]
    locals()["fine_grained_gru_"+str(i)] = pd.read_csv("fine_grained_gru_"+str(i)+".csv").iloc[:,range(1,6)]
    locals()["fine_grained_lstm_"+str(i)]= pd.read_csv("fine_grained_lstm_"+str(i)+".csv").iloc[:,range(1,6)]
 
# Initialize the data frames from each model to zero
binary_gru = pd.DataFrame(0, index=np.arange(11), columns=['time','train_loss','train','dev','test'])
binary_lstm = pd.DataFrame(0, index=np.arange(11), columns=['time','train_loss','train','dev','test'])
fine_grained_gru = pd.DataFrame(0, index=np.arange(11), columns=['time','train_loss','train','dev','test'])
fine_grained_lstm = pd.DataFrame(0, index=np.arange(11), columns=['time','train_loss','train','dev','test'])
 
# Populate the dataframes
for i in xrange(11):
    for j in xrange(5):
        if np.count_nonzero(np.isnan([binary_gru_1.iloc[i,j],binary_gru_2.iloc[i,j],binary_gru_3.iloc[i,j],binary_gru_4.iloc[i,j]])) == 4: # if all are NAs
            binary_gru.iloc[i,j] = 'nan'
        elif np.count_nonzero(np.isnan([binary_gru_1.iloc[i,j],binary_gru_2.iloc[i,j],binary_gru_3.iloc[i,j],binary_gru_4.iloc[i,j]])) > 0: # if at least oen NA 
            binary_gru.iloc[i,j] = np.nansum([binary_gru_1.iloc[i,j],binary_gru_2.iloc[i,j],binary_gru_3.iloc[i,j],binary_gru_4.iloc[i,j]])/(4-np.count_nonzero(np.isnan([binary_gru_1.iloc[i,j],binary_gru_2.iloc[i,j],binary_gru_3.iloc[i,j],binary_gru_4.iloc[i,j]])))
        else :
            binary_gru.iloc[i,j] = (binary_gru_1.iloc[i,j]+binary_gru_2.iloc[i,j]+binary_gru_3.iloc[i,j]+binary_gru_4.iloc[i,j])/4      
        if np.count_nonzero(np.isnan([binary_lstm_1.iloc[i,j],binary_lstm_2.iloc[i,j],binary_lstm_3.iloc[i,j],binary_lstm_4.iloc[i,j]])) == 4: # if all are NAs
            binary_lstm.iloc[i,j] = 'nan'
        elif np.count_nonzero(np.isnan([binary_lstm_1.iloc[i,j],binary_lstm_2.iloc[i,j],binary_lstm_3.iloc[i,j],binary_lstm_4.iloc[i,j]])) > 0: # if at least oen NA 
            binary_lstm.iloc[i,j] = np.nansum([binary_lstm_1.iloc[i,j],binary_lstm_2.iloc[i,j],binary_lstm_3.iloc[i,j],binary_lstm_4.iloc[i,j]])/(4-np.count_nonzero(np.isnan([binary_lstm_1.iloc[i,j],binary_lstm_2.iloc[i,j],binary_lstm_3.iloc[i,j],binary_lstm_4.iloc[i,j]])))
        else :
            binary_lstm.iloc[i,j] = (binary_lstm_1.iloc[i,j]+binary_lstm_2.iloc[i,j]+binary_lstm_3.iloc[i,j]+binary_lstm_4.iloc[i,j])/4    
        if np.count_nonzero(np.isnan([fine_grained_gru_1.iloc[i,j],fine_grained_gru_2.iloc[i,j],fine_grained_gru_3.iloc[i,j],fine_grained_gru_4.iloc[i,j],fine_grained_gru_5.iloc[i,j]])) == 5: # if all are NAs
            fine_grained_gru.iloc[i,j] = 'nan'
        elif np.count_nonzero(np.isnan([fine_grained_gru_1.iloc[i,j],fine_grained_gru_2.iloc[i,j],fine_grained_gru_3.iloc[i,j],fine_grained_gru_4.iloc[i,j],fine_grained_gru_5.iloc[i,j]])) > 0: # if at least oen NA 
            fine_grained_gru.iloc[i,j] = np.nansum([fine_grained_gru_1.iloc[i,j],fine_grained_gru_2.iloc[i,j],fine_grained_gru_3.iloc[i,j],fine_grained_gru_4.iloc[i,j],fine_grained_gru_5.iloc[i,j]])/(5-np.count_nonzero(np.isnan([fine_grained_gru_1.iloc[i,j],fine_grained_gru_2.iloc[i,j],fine_grained_gru_3.iloc[i,j],fine_grained_gru_4.iloc[i,j],fine_grained_gru_5.iloc[i,j]])))
        else :
            fine_grained_gru.iloc[i,j] = (fine_grained_gru_1.iloc[i,j]+fine_grained_gru_2.iloc[i,j]+fine_grained_gru_3.iloc[i,j]+fine_grained_gru_4.iloc[i,j]+fine_grained_gru_5.iloc[i,j])/5    
        if np.count_nonzero(np.isnan([fine_grained_lstm_1.iloc[i,j],fine_grained_lstm_2.iloc[i,j],fine_grained_lstm_3.iloc[i,j],fine_grained_lstm_4.iloc[i,j],fine_grained_lstm_5.iloc[i,j]])) == 5: # if all are NAs
            fine_grained_lstm.iloc[i,j] = 'nan'
        elif np.count_nonzero(np.isnan([fine_grained_lstm_1.iloc[i,j],fine_grained_lstm_2.iloc[i,j],fine_grained_lstm_3.iloc[i,j],fine_grained_lstm_4.iloc[i,j],fine_grained_lstm_5.iloc[i,j]])) > 0: # if at least oen NA 
            fine_grained_lstm.iloc[i,j] = np.nansum([fine_grained_lstm_1.iloc[i,j],fine_grained_lstm_2.iloc[i,j],fine_grained_lstm_3.iloc[i,j],fine_grained_lstm_4.iloc[i,j],fine_grained_lstm_5.iloc[i,j]])/(5-np.count_nonzero(np.isnan([fine_grained_lstm_1.iloc[i,j],fine_grained_lstm_2.iloc[i,j],fine_grained_lstm_3.iloc[i,j],fine_grained_lstm_4.iloc[i,j],fine_grained_lstm_5.iloc[i,j]])))
        else :
            fine_grained_lstm.iloc[i,j] = (fine_grained_lstm_1.iloc[i,j]+fine_grained_lstm_2.iloc[i,j]+fine_grained_lstm_3.iloc[i,j]+fine_grained_lstm_4.iloc[i,j]+fine_grained_lstm_5.iloc[i,j])/5
 
# Create the test data frame with mean accuracy                      
test = pd.DataFrame(0,index = ['gru','lstm'],columns = ['Binary','Fine_grained'])
test.iloc[0,0] = binary_gru.iloc[1,4]     
test.iloc[1,0] = binary_lstm.iloc[1,4]
test.iloc[0,1] = fine_grained_gru.iloc[1,4]
test.iloc[1,1] = fine_grained_lstm.iloc[1,4]
#Check out the variance
test_sd = pd.DataFrame(0,index = ['gru','lstm'],columns = ['Binary','Fine_grained'])
test_sd.iloc[0,0] = np.std([binary_gru_1.iloc[:,4].max(),binary_gru_2.iloc[:,4].max(),binary_gru_3.iloc[:,4].max(),binary_gru_4.iloc[:,4].max()])
test_sd.iloc[1,0] = np.std([binary_lstm_1.iloc[:,4].max(),binary_lstm_2.iloc[:,4].max(),binary_lstm_3.iloc[:,4].max(),binary_lstm_4.iloc[:,4].max()])
test_sd.iloc[0,1] = np.std([fine_grained_gru_1.iloc[:,4].max(),fine_grained_gru_2.iloc[:,4].max(),fine_grained_gru_3.iloc[:,4].max(),fine_grained_gru_4.iloc[:,4].max(),fine_grained_gru_5.iloc[:,4].max()])
test_sd.iloc[1,1] = np.std([fine_grained_lstm_1.iloc[:,4].max(),fine_grained_lstm_2.iloc[:,4].max(),fine_grained_lstm_3.iloc[:,4].max(),fine_grained_lstm_4.iloc[:,4].max(),fine_grained_lstm_5.iloc[:,4].max()])
 
print test
print test_sd
#########    Visualize the results  ############
# Average Time 
# Binary Classification
plt.plot(binary_gru.iloc[:,0] ,'r-' ,label = "Tree-GRU")
plt.plot(binary_lstm.iloc[:,0] , 'g-', label = "Tree-LSTM")
plt.title('Binary Average time')
plt.legend(loc='upper left ')
plt.show()
 
# Fine_grained Classification
plt.plot(fine_grained_gru.iloc[:,0] ,'r-' ,label = "Tree-GRU")
plt.plot(fine_grained_lstm.iloc[:,0] , 'g-', label = "Tree-LSTM")
plt.title('Binary Average time')
plt.legend(loc='upper left ')
plt.show()
 
# Average Loss
 
plt.plot(binary_gru.iloc[:,1] ,'r-' ,label = "Tree-GRU")
plt.plot(binary_lstm.iloc[:,1] , 'g-', label = "Tree-LSTM")
plt.title('Binary Average time')
plt.legend()
plt.show()
 
# Fine_grained Classification
plt.plot(fine_grained_gru.iloc[:,1] ,'r-' ,label = "Tree-GRU")
plt.plot(fine_grained_lstm.iloc[:,1] , 'g-', label = "Tree-LSTM")
plt.title('Binary Average time')
plt.legend()
plt.show()
 
# Trarining process
# Binary Classification
 
plt.plot(binary_gru.iloc[:,2] ,'r-' ,label = "Training")
plt.plot(binary_gru.iloc[:,3] , 'g-', label = "Validation")
plt.title('Tree-GRU Training Process')
plt.legend(loc='upper left')
plt.show()
  
plt.plot(binary_lstm.iloc[:,2] ,'r-' ,label = "Training")
plt.plot(binary_lstm.iloc[:,3] , 'g-', label = "Validation")
plt.title('Tree-LSTM Training Process')
plt.legend(loc='upper left')
plt.show()
 
# Fine grained Classification
 
plt.plot(fine_grained_gru.iloc[:,2] ,'r-' ,label = "Training")
plt.plot(fine_grained_gru.iloc[:,3] , 'g-', label = "Validation")
plt.title('Tree-GRU Training Process')
plt.legend(loc='upper left')
plt.show()
  
plt.plot(fine_grained_lstm.iloc[:,2] ,'r-' ,label = "Training")
plt.plot(fine_grained_lstm.iloc[:,3] , 'g-', label = "Validation")
plt.title('Tree-LSTM Training Process')
plt.legend(loc='upper left')
plt.show()
