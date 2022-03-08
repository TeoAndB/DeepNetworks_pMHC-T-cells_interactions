# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:34:35 2021

@author: bejen
"""
from tensorflow import keras
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelSpreading
from tensorflow.keras.models import load_model
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

path = './results/'
df_setup1 = pd.read_excel(path + 'pred_results_netTCR1_allTCRs_lr0_001.xlsx')
df_setup2_1 = pd.read_excel(path + 'pred_results_netTCR2_allTCRs_lr0_001.xlsx')
df_setup2_2 = pd.read_excel(path + 'pred_results_netTCR2_allTCRs_lr0_01.xlsx')


#%%
# Confusion Matrix Setup 1: 
true_labels = df_setup1['True binder'].tolist()
pl_vector =  df_setup1['Prediction result'].to_numpy()
predicted_labels =  np.zeros(len(true_labels))
predicted_labels = (pl_vector >0.5).astype(int).tolist()

cm1 = confusion_matrix(true_labels, predicted_labels)

#%%
# Confusion Matrix Setup 1: 
true_labels = df_setup2_1['True binder'].tolist()
pl_vector =  df_setup2_1['Prediction result'].to_numpy()
predicted_labels =  np.zeros(len(true_labels))
predicted_labels = (pl_vector >0.5).astype(int).tolist()

cm2_1 = confusion_matrix(true_labels, predicted_labels)

#%%
# Confusion Matrix Setup 1: 
true_labels = df_setup2_2['True binder'].tolist()
pl_vector =  df_setup2_2['Prediction result'].to_numpy()
predicted_labels =  np.zeros(len(true_labels))
predicted_labels = (pl_vector >0.5).astype(int).tolist()

cm2_2 = confusion_matrix(true_labels, predicted_labels)


#%% Plotting:
# NB: RUN EACH CELL AT A TIME

df_cm1 = pd.DataFrame(cm1, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm1, annot=True, fmt="d", cmap="YlGnBu")

  
#%%
df_cm2 = pd.DataFrame(cm2_1, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm2, annot=True, fmt="d", cmap="YlGnBu")

#%%
df_cm3 = pd.DataFrame(cm2_2, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm3, annot=True, fmt="d", cmap="YlGnBu")
