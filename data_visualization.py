# -*- coding: utf-8 -*-
'''
Data visualization of MIRA data sets: peptide-detail-ci.csv (based on subject-metadata.csv)
@author: Andreea Teodora Bejenariu

The following code will provide:
1. An overview of:
    - the distribution of cell types 
    - the distribution of cohort types
    For the data set before (peptide-detail-ci.csv) and after processing (the data set of poisitives)
    
2. Whether the data set of negatives contains a balanced number of TCRs and peptides
'''
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
# CHANGE here: name of the original and processed data sets

path = './data/'
name_original_dataset = 'peptide-detail-ci.csv'
subject_metadata_filename = 'subject-metadata.csv'

# Name of the processed data files
path_res = './processet_data_results/'

# data set with positives only
name_dataset_wpositives = 'peptide_dataset_wHLA_positives.xlsx' # change here
# data set with positives and negatives
name_dataset_pos_neg = 'peptide_dataset_wHLA_pos_neg.xlsx' # change here

#%%
# Reading the original data sets
peptide_detail_ci = pd.read_csv(path + name_original_dataset)
subject_metadata = pd.read_csv(path + subject_metadata_filename,encoding= 'unicode_escape')

# Merge peptide_detail_ci and peptide_detail_cii
peptide_detail = peptide_detail_ci

# Keep columns of interest
peptide_detail_c = peptide_detail[['TCR BioIdentity','Experiment','Amino Acids']]

# Rename columns:
peptide_detail_c = pd.DataFrame(peptide_detail_c)

#%%
# Readign the processed data sets
df_pos =  pd.read_excel(path_res + name_dataset_wpositives)
df_pos_neg = pd.read_excel(path_res + name_dataset_pos_neg)

#%%
# ORIGINAL DATA SET: Checking data set distribution based on 'Cell Type' and and 'Cohort' in subject-metadata-ci.csv 
print('Data visualtisation plots for the orginal data set. In progress...')

peptide_detail_c2 = peptide_detail_c.copy()

peptide_detail_c2['Cell Type'] = np.NaN
peptide_detail_c2['Cohort'] = np.NaN
peptide_detail_c2['HLA-A col 1'] = np.NaN
peptide_detail_c2['HLA-A col 2'] = np.NaN

# keep columns of interest
subject_metadata_original = subject_metadata.copy()
subject_metadata_original = subject_metadata_original[['Experiment','Cell Type','Cohort', 'HLA-A', 'HLA-A.1']]

#Convert dataframes to numpy
peptide_details_np = peptide_detail_c2.to_numpy()
subject_metadata_np = subject_metadata_original.to_numpy()

np_pep = peptide_details_np.copy()
row_peptide_details = 0

# Map Cell Type and Cohort values based on the matching Experiment values:
for exp_id in (np_pep[:,1]):
    exp_id1 = exp_id
    index_arr = (np.where(subject_metadata_np[:,0]==exp_id1))[0]
    row_metadata = index_arr[0] #access row index of metadata
    val_cell_type = subject_metadata_np[row_metadata,1] #Access value of Cell Type
    val_cohort = subject_metadata_np[row_metadata,2] #Access value of Cohort
    val_hla_1 = subject_metadata_np[row_metadata,3] # access HLA-A value 1
    val_hla_2 = subject_metadata_np[row_metadata,4] # access HLA-A value 2
    np_pep[row_peptide_details,3] = val_cell_type #update value for Cell Type column at specified index in np_pep
    np_pep[row_peptide_details,4] = val_cohort #update value for Cohort column at specified index in np_pep
    np_pep[row_peptide_details,5] = val_hla_1 #update HLA-A value 1
    np_pep[row_peptide_details,6] = val_hla_2
    row_peptide_details +=1

# Plotting bar charts
df = pd.DataFrame(np_pep, columns = peptide_detail_c2.columns)

# For HLA types:
sns.set(style="whitegrid")
plt.figure(figsize=(20,5))
total = float(len(df))
ax = sns.countplot(x="HLA-A col 1", hue="HLA-A col 1", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.legend(bbox_to_anchor=(1, 0.1))
plt.title('peptide-detail-ci.csv - distribution based on HLA-A column 1', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')    
    current_width = p.get_width()
    diff = current_width - 0.55
        # we change the bar width
    p.set_width(0.55)
        # recenter the bar
    p.set_x(p.get_x() + diff * .5)
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(20,5))
total = float(len(df))
ax = sns.countplot(x="HLA-A col 2", hue="HLA-A col 2", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.legend(bbox_to_anchor=(1, 1.2))
plt.title('peptide-detail-ci.csv - distribution based on HLA-A column 2', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')     
    current_width = p.get_width()
    diff = current_width - 0.55
        # change the bar width
    p.set_width(0.55)
        # recenter the bar
    p.set_x(p.get_x() + diff * .5)
plt.show()
    
# For Cell Type
sns.set(style="whitegrid")
plt.figure(figsize=(3,5))
total = float(len(df))
ax = sns.countplot(x="Cell Type", hue="Cell Type", data=df)
plt.title('peptide-detail-ci.csv - distribution based on cell type', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.show()


# For Cohorts
sns.set(style="whitegrid")
plt.figure(figsize=(7,5))
plt.tight_layout()
total = float(len(df))
ax = sns.countplot(x="Cohort", hue="Cohort", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title('peptide-detail-ci.csv - distribution based on cohort type', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
    current_width = p.get_width()
    diff = current_width - 0.55
        # change the bar width
    p.set_width(0.55)
        # recenter the bar
    p.set_x(p.get_x() + diff * .5)
plt.show()

print('Data visualtisation plots for the orginal data set. Done\n')

#%%

# DATA SET OF POSITIVES: Checking data set distribution based on 'Cell Type' and and 'Cohort' in the data set of positives

print('Data visualtisation plots for the data set of positives. In progress...')

df_pep_data_check = df_pos.copy()
subject_metadata_original = subject_metadata.copy()

df_pep_data_check['Cell Type'] = np.NaN
df_pep_data_check['Cohort'] = np.NaN

# keep columns of interest
subject_metadata_original = subject_metadata_original[['Experiment','Cell Type','Cohort']]

#Convert dataframes to numpy
peptide_details_np = df_pep_data_check.to_numpy()
subject_metadata_np = subject_metadata_original.to_numpy()

np_pep = peptide_details_np.copy()
row_peptide_details = 0

# Map Cell Type and Cohort values based on the matching Experiment values
for exp_id in (np_pep[:,1]):
    exp_id1 = exp_id
    index_arr = (np.where(subject_metadata_np[:,0]==exp_id1))[0]
    row_metadata = index_arr[0] #access row index of metadata
    val_cell_type = subject_metadata_np[row_metadata,1] #Access value of Cell Type
    val_cohort = subject_metadata_np[row_metadata,2] #Access value of Cohort
    np_pep[row_peptide_details,6] = val_cell_type #update value for Cell Type column at specified index in np_pep
    np_pep[row_peptide_details,7] = val_cohort #update value for Cohort column at specified index in np_pep
    row_peptide_details +=1

# Plotting bar charts
df = pd.DataFrame(np_pep, columns = df_pep_data_check.columns)


# For Cell Type
sns.set(style="whitegrid")
plt.figure(figsize=(3,5))
total = float(len(df))
ax = sns.countplot(x="Cell Type", hue="Cell Type", data=df)
plt.title('Data set of positives - distribution based on cell type', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.show()


# For Cohorts
sns.set(style="whitegrid")
plt.figure(figsize=(1,5))
total = float(len(df))
ax = sns.countplot(x="Cohort", hue="Cohort", data=df)
ax.legend(bbox_to_anchor=(4.6, 0.8))
plt.title('Data set of positives - distribution based on cohort type', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.show()

print('Data visualtisation plots for the data set of positives. Done.\n')
#%%
# DATA SET OF POSITIVES AND NEGATIVES: Quick checking whther the data set is balanced: 


df_pep_data_pos_neg = df_pos_neg.copy()
df_pep_data_positives = df_pep_data_pos_neg[df_pep_data_pos_neg.binding_score == 1]
df_pep_data_negatives = df_pep_data_pos_neg[df_pep_data_pos_neg.binding_score == 0]

# Checks whether each CDR3 occurs exactly twice: 
col_tcr = df_pep_data_pos_neg['CDR3'].value_counts() 
x = col_tcr.values.tolist()
#print(x.count(2) == len(x)) 
print('\nStatement: Each CDR3 occur exactly once in the data set of positives and negatives - ', x.count(2) == len(x)) 

# Checks whether each peptide has the same occurence in the data set of positives and data set of negatives:
col_peptide_pos = df_pep_data_positives['Peptide_antigen'].value_counts()
print('\nPeptide counts in the data set of positives\n',col_peptide_pos)

col_peptide_neg = df_pep_data_negatives['Peptide_antigen'].value_counts()
print('\nPeptide counts in the data set of negatives\n',col_peptide_neg)

#print(col_peptide_neg.tolist() == col_peptide_pos.tolist())
print('\nStatement: Each peptide has the same occurence in data set of positives and data set of negatives - ', x.count(2) == len(x)) 


# VISUALISATION ##############
print('\nData visualtisation plot for the data set of positives and negatives. In progress...')

df_pep_pos = col_peptide_pos.to_frame().rename(columns={"Peptide_antigen": "Peptide counts in positives"})
df_pep_pos['Peptide'] = df_pep_pos.index
df_pep_pos['Peptide counts plus negatives'] = col_peptide_pos + col_peptide_neg

# Plotting:
sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Plot the peptides from the positives
sns.set_color_codes("pastel")
sns.barplot(x="Peptide counts plus negatives", y="Peptide", data=df_pep_pos,
            label="Peptide counts in the data set of positives", color="b")


# Plot the peptides where the negatives were added
sns.set_color_codes("muted")
sns.barplot(x="Peptide counts in positives", y="Peptide", data=df_pep_pos,
            label="Peptide counts in the data set of negatives", color="b")
  

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.legend(bbox_to_anchor=(1, 1.05))
ax.set(ylabel="",
       xlabel="Peptide distribution between the data set of positives and data set of negatives")
sns.despine(left=True, bottom=True)

print('Data visualtisation plot for the data set of positives and negatives. Done.')
