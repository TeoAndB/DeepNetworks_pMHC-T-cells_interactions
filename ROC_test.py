# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:14:55 2021

@author: bejen
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt





path = './results/'
df_setup2_1 = pd.read_excel(path + 'pred_results_netTCR2_allTCRs_lr0_001.xlsx')
df_setup2_2 = pd.read_excel(path + 'pred_results_netTCR2_allTCRs_lr0_01.xlsx')
df_setup1 = pd.read_excel(path + 'pred_results_netTCR1_allTCRs_lr0_001.xlsx')




fpr1 , tpr1, thresholds1 = roc_curve(df_setup1['True binder'], df_setup1['Prediction result'])
auc1 = metrics.roc_auc_score(df_setup1['True binder'], df_setup1['Prediction result'])

fpr2 , tpr2, thresholds2 = roc_curve(df_setup2_1['True binder'], df_setup2_1['Prediction result'])
auc2 = metrics.roc_auc_score(df_setup2_1['True binder'], df_setup2_1['Prediction result'])

fpr3 , tpr3, thresholds3 = roc_curve(df_setup2_2['True binder'], df_setup2_2['Prediction result'])
auc3 = metrics.roc_auc_score(df_setup2_2['True binder'], df_setup2_2['Prediction result'])

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, label= 'Setup 1 (AUC = %0.2f)' % auc1)
plt.plot(fpr2, tpr2, label= 'Setup 2.1 (AUC = %0.2f)' % auc2)
plt.plot(fpr3, tpr3, label= 'Setup 2.2 (AUC = %0.2f)' % auc3)
plt.legend(loc=4, prop={'size': 12})
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC Curves and AUC scores for the 3 NetTCR setups')
plt.show()
