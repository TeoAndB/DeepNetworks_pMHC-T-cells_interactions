# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:00:58 2021

@author: bejen
"""

import numpy as np
import data_encoding as encoding
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import CNN_NetTCR_2 as CNN   #CNN model imported: choose between CNN_NetTCR_1 or CNN_NetTCR_2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.font_manager import FontProperties

#%%
# Reading in the data

path = './data/'

data_filename = 'train_covid_data_red94_part90.xlsx'
blosum_filename = 'BLOSUM62_matrix.txt'

df = pd.read_excel(path + data_filename)
print("Creating BLOSUM dictionary...")
blosum_dict = encoding.give_blosum_dict(path + blosum_filename)
print("BlOSUM dictionary created")

#%%
'''
Partitioning the data:

    CV_TCR_list - list of encoded tcr for each partition
    CV_pep_list - list of encoded peptide for each partition
    CV_y_list - list of binders for each partition
    
'''
df = df.iloc[:,:5]

# TCR
data_tcr_np = df.iloc[:,:4].to_numpy()
n_max_tcr = df.CDR3.str.len().max()
CV_TCR_Xlist = []

# Peptide:
df_pep = df.iloc[:,1:5]
df_pep = df_pep[['peptide','allele','binder','partition']]
data_pep_np = df_pep.to_numpy()

CV_pep_Xlist = []

CV_y_list = []

df_temp = df.loc[df['binder'] == 1]
df2 = df_temp[['CDR3','peptide','allele','partition']]
df2 = df2.set_index('CDR3').rename_axis(index=(None), columns=None)
tcr_dict = df2.to_dict('index')

TCR_pep_list = []
    
for i in range(1,6):
    #TCR:    
    CV_TCR = np.delete(data_tcr_np[data_tcr_np[:,3]==i], [1,3], axis = 1)
    CV_TCR_list = list(CV_TCR[:,0])
    print("Encoding TCRs - partition {}/{}...".format(i,5))
    CV_tcr_matrix_data = encoding.matrix_encoding_per_TCR_list(CV_TCR_list, n_max_tcr, blosum_dict)
    CV_TCR_Xlist.append(CV_tcr_matrix_data)
    
    #Peptide:
    CV_pep = np.delete(data_pep_np[data_pep_np[:,3]==i], [1,3], axis = 1)
    CV_pep_list = list(CV_pep[:,0])
    print("Encoding peptides - partition {}/{}...".format(i,5))
    CV_pep_matrix_data = encoding.matrix_encoding_per_peptide_list(CV_pep_list, blosum_dict)
    CV_pep_Xlist.append(CV_pep_matrix_data)
    
    #y - same for TCR and peptide:
    CV_y = list(CV_pep[:,1])
    CV_y_list.append(CV_y)
    
    #list of TCR, peptides, HLA, partition and binders
    # the indices will correspond with the predicted result list: pred_result_list
    part_list =  np.delete(data_pep_np[data_pep_np[:,3]==i], [0,1,2], axis = 1).flatten().tolist()
    HLA_list =  np.delete(data_pep_np[data_pep_np[:,3]==i], [0,2,3], axis = 1).flatten().tolist()
    
    TCR_pep_list.append([CV_TCR_list, CV_pep_list, HLA_list, part_list, CV_y])

#%%

# Nested Model
# Dictionary to remember partitions: key: Model no. - values: (test partition, validation partition)
partition_dict = { 'm1_1': ('t1','v2'), 'm1_2': ('t1','v3'), 'm1_3': ('t1','v4'), 'm1_4': ('t1','v5'),
                   'm2_1': ('t2','v1'), 'm2_3': ('t2','v3'), 'm2_4': ('t2','v4'), 'm2_5': ('t2','v5'),
                   'm3_1': ('t3','v1'), 'm3_2': ('t3','v2'), 'm3_4': ('t3','v4'), 'm3_5': ('t3','v5'),
                   'm4_1': ('t4','v1'), 'm4_2': ('t4','v2'), 'm4_3': ('t4','v3'), 'm4_5': ('t4','v5'),
                   'm5_1': ('t5','v1'), 'm5_2': ('t5','v2'), 'm5_3': ('t5','v3'), 'm5_4': ('t5','v4')
                    }

#%%
# Simple model first

tf.random.set_seed(0)

model_data = partition_dict.copy() # CHANGE HERE WITH partition_dict

#replace key values with datasets
for key in partition_dict:
    t_set = int(partition_dict[key][0][1]) - 1
    v_set = int(partition_dict[key][1][1]) - 1
    
    indices = [0,1,2,3,4]
    tcr_train_list = []
    pep_train_list = []
    y_train_list = []
    for i in indices:
        if i != t_set and i != v_set:
            tcr_train_list.append(CV_TCR_Xlist[i])
            pep_train_list.append(CV_pep_Xlist[i])
            y_train_list.append(CV_y_list[i]) 
    
    
    CV_TCR_Xtrain = np.concatenate((tcr_train_list[0],tcr_train_list[1],tcr_train_list[2]), axis = 0)
    CV_pep_Xtrain = np.concatenate((pep_train_list[0],pep_train_list[1],pep_train_list[2]), axis = 0)
    CV_y_train = np.concatenate((y_train_list[0],y_train_list[1],y_train_list[2]), axis = 0)
    
    CV_TCR_Xtest = CV_TCR_Xlist[t_set]
    CV_pep_Xtest = CV_pep_Xlist[t_set]
    CV_y_test = np.array(CV_y_list[t_set])
    
    
    
    CV_TCR_Xvalid = CV_TCR_Xlist[v_set]
    CV_pep_Xvalid = CV_pep_Xlist[v_set]
    CV_y_valid = np.array(CV_y_list[v_set])
    
    model_data[key] = ((CV_TCR_Xtrain, CV_pep_Xtrain, CV_y_train),
                       (CV_TCR_Xtest,CV_pep_Xtest, CV_y_test),
                       (CV_TCR_Xvalid,CV_pep_Xvalid, CV_y_valid))



#%%

# MODEL HYPERPARAMETERS - CHANGE HERE:

l_rate = 0.001 #learning rate for Adam optimizer
no_filters = 16 #no of convolutional filters to be used
hidden_units = 100 #no of hidden units

#%%

# TRAINING AND VALIDATING THE MODEL:

pred_result = []
pred_result_list = []


# For plotting
fig, ax = plt.subplots(5,4, sharex = True, sharey = True, figsize=(15,8)) #CHANGE
fig.tight_layout(pad=1.8)
fig.subplots_adjust(top=0.90)
fontP = FontProperties()
fontP.set_size('medium')
fig.suptitle('Monitoring Loss', fontsize=16)
plt.setp(ax[-1, :], xlabel='Epoch')
plt.setp(ax[:, 0], ylabel='Loss')

# Building the model:
tf.random.set_seed(0)

counter = 1
for key in model_data:
    print('Training model:', key)
    
    model_i = model_data[key]
    
    # Training data
    CV_TCR_Xtrain = (model_i[0][0]).astype(np.float32)
    CV_pep_Xtrain = (model_i[0][1]).astype(np.float32)
    CV_ytrain = (model_i[0][2]).astype(np.float32)
    
    # Validation Data
    CV_TCR_Xtest = (model_i[1][0]).astype(np.float32)
    CV_pep_Xtest = (model_i[1][1]).astype(np.float32)
    CV_ytest = (model_i[1][2]).astype(np.float32)
    
    # Testing Data
    CV_TCR_Xvalid = (model_i[2][0]).astype(np.float32)
    CV_pep_Xvalid = (model_i[2][1]).astype(np.float32)
    CV_yvalid = (model_i[2][2]).astype(np.float32)
    
    tf.keras.backend.clear_session()
    
    x_tcr, x_pep, output = CNN.build_CNN(n_filters=no_filters, n_hidden=hidden_units)
    
    model = keras.Model(
        inputs=[x_tcr, x_pep],
        outputs=[output],
        )
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=l_rate))
    
    earlystop = EarlyStopping(
                    monitor='val_loss', patience=30, verbose = 1,  #verbose = "max"
                    mode='min', restore_best_weights=True)
    
    # LOOP - per each model - one ensemble model predicion based on 4 models
    history = model.fit(x=[CV_TCR_Xtrain, CV_pep_Xtrain],y=CV_ytrain,
              validation_data=([CV_TCR_Xvalid,CV_pep_Xvalid],CV_yvalid),
              batch_size=128, epochs=300, verbose=1,
              callbacks=[earlystop])
    
    #filepath = 'C:/Users/bejen/OneDrive/Desktop/DTU_lectures/6SEM/Bachelor_thesis/data_processing/best_models/'  
    filepath = "./best_models/"
    model_name = 'model_M' + key[1] +'_v' + partition_dict[key][0][1] + '_t' + partition_dict[key][1][1] + '.h5'
    model.save(filepath + model_name)
    
    # Plotting loss for each model:
    if counter <= 4: m_no = 0 
    elif counter > 4 and counter <=8: m_no = 1 
    elif counter >8 and counter <=12: m_no = 2
    elif counter >12 and counter <=16 : m_no = 3
    elif  counter>16 : m_no = 4 
    ax_no = 3 if counter%4==0 else ((counter%4)-1)
    
    title_plot = 'Model M'+ key[1] +'_v' + partition_dict[key][0][1] + '_t' + partition_dict[key][1][1]
    ax[m_no, ax_no-1].set_title(title_plot, fontsize=10)
    ax[m_no, ax_no].plot(history.history['loss'], label='train')
    ax[m_no, ax_no].plot(history.history['val_loss'], label='validation')
    ax[0,0].legend(title='Legend', bbox_to_anchor=(4.55, 1), 
                   loc='upper left', prop=fontP)  

    
    y_pred = model.predict([CV_TCR_Xtest, CV_pep_Xtest]).ravel()
    
    pred_result.append(y_pred)

    
    if len(pred_result) == 4:
        pred_result_np = np.asarray(pred_result)
        pred_rezult_avg = np.average(pred_result_np, axis=0)
        pred_result_list.append(pred_rezult_avg)
        pred_result = []
    
    counter +=1
   
    
fig.savefig('loss_monitor_netTCR2_lr0_001.png')
# Match TCR_pep_list with pred_result_list - creating a data frame with the results
for i in range(len(pred_result_list)):
    TCR_pep_list[i].append(list(pred_result_list[i]))
  
tcr_col, pep_col, HLA_col, part_col, true_binder_col, pred_result = ([] for i in range(6))

for i in range(5):
    tcr_col.extend(TCR_pep_list[i][0])
    pep_col.extend(TCR_pep_list[i][1])
    HLA_col.extend(TCR_pep_list[i][2])
    part_col.extend(TCR_pep_list[i][3])
    true_binder_col.extend(TCR_pep_list[i][4])
    pred_result.extend(TCR_pep_list[i][5])
    
df_results = pd.DataFrame(list(zip(tcr_col, pep_col, HLA_col, part_col, true_binder_col, pred_result)),
               columns =['CDR3', 'Peptide', 'Allele', 'Partition', 'True binder', 'Prediction result'])

df_results.to_excel('pred_results_netTCR2_lr0_001.xlsx')


#%%
# PERFORMANCE METRICS: AUC

roc_auc_score(true_binder_col, pred_result, multi_class='ovr')

fpr, tpr, _ = roc_curve(true_binder_col, pred_result)
roc_auc = auc(fpr, tpr)
# Plot val loss per model
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_netTCR2_lr0_001.png')
plt.show()


