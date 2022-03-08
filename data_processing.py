# -*- coding: utf-8 -*-
"""
Data processing of MIRA data sets: peptide-detail-ci.csv and subject-metadata.csv
    - all T cells considered

@author: Andreea Teodora Bejenariu
"""

import pandas as pd
import numpy as np
import operator
import random

#%%
# Read data
# NB: Observations for this dataset, Feb 18 13:57:52 2021: peptide_detail_ci: 154320
path = './data/'
peptide_detail_ci = pd.read_csv(path + 'peptide-detail-ci.csv')
subject_metadata = pd.read_csv(path + 'subject-metadata.csv',encoding= 'unicode_escape')

# Merge peptide_detail_ci and peptide_detail_cii
peptide_detail = peptide_detail_ci

# Keep columns of interest
peptide_detail_c = peptide_detail[['TCR BioIdentity','Experiment','Amino Acids']]

# Rename columns:
peptide_detail_reNamed = peptide_detail_c.rename(columns={"TCR BioIdentity": "TCR_beta", "Amino Acids": "Antigen_AA"})

print('Kept columns of interest and renamed them: TCR_beta, Experiment, Antigen_AA \nObservations: {}\n'.format(peptide_detail_reNamed.shape[0]))

#%%

peptide_dettail_AA1 = peptide_detail_reNamed.copy()
# Get a list of the columns
columns_titles = ["TCR_beta","Experiment","Antigen_AA"]
peptide_dettail_AA1=peptide_dettail_AA1.reindex(columns=columns_titles)

peptide_dettail_AA2 = peptide_dettail_AA1.Antigen_AA.str.split(",",expand=True)
peptide_dettail_AA2=peptide_dettail_AA2.dropna(axis=1,how='all')
col_pep = ['pep1','pep2', 'pep3', 'pep4','pep5','pep6', 'pep7', 'pep8','pep9','pep10','pep11','pep12','pep13']
peptide_dettail_AA2.columns = col_pep

peptide_detail_expanded = pd.concat([peptide_dettail_AA1.iloc[:,0:2],peptide_dettail_AA2], axis = 1)

# Convert to long 
peptide_detail_double = pd.melt(peptide_detail_expanded, id_vars = ['TCR_beta','Experiment'], value_vars = col_pep)
peptide_detail_double = peptide_detail_double.drop(['variable'], axis = 1)
peptide_detail_double = peptide_detail_double.rename(columns={"value": "Antigen_AA"})
peptide_detail_double.iloc[0,:]

# Drop NaN values for Antigen_AA:
peptide_detail_double = peptide_detail_double.dropna(axis = 0)
peptide_detail_double= peptide_detail_double.reset_index(drop=True)
print('Split peptide observations (Antigen_AA) into long \nObservations: {}\n'.format(peptide_detail_double.shape[0]))

#%%

# Process TCR_beta column: remove V and J gene-strings - select only the CDR3 region  
peptide_detail_TCRb = pd.DataFrame(peptide_detail_double.TCR_beta.str.split('+',1).tolist(),
                                 columns = ['CDR3','VJ_extras'])
peptide_detail_TCRb = pd.concat([peptide_detail_TCRb['CDR3'],peptide_detail_double['Experiment'],peptide_detail_double['Antigen_AA']], axis = 1)

print('Removed V and J gene-strings from TCR_beta column. New column names: CDR3, Experiment, Antigen_AA \n ')

#%%
# Remove rows which do not have standard aminoacids for the antigen sequence

standard_aa = 'ARNDCQEGHILKMFPSTWYV'
def compare_aa(peptide):
    """Returns peptide if it contains standard amino acids

    Parameters
    ----------
    peptide : str
        Peptide input to check whether all amino acids are standard

    Returns
    -------
    peptide: str
        Peptide remains unchanged if all amino acids are standard. If not, peptide is an empty string
    """
    for i in peptide:
        if i not in standard_aa:
             peptide = ''
    return peptide

g = np.vectorize(compare_aa, otypes=[pd.core.series.Series]) #vectorizing the function

peptide_detail_TCRb_2 = peptide_detail_TCRb.copy()
peptide_detail_TCRb_2['Antigen_AA'] = g(peptide_detail_TCRb_2['Antigen_AA'])
peptide_detail_TCRb_2['Antigen_AA'].replace('', np.nan, inplace=True)
peptide_detail_TCRb_2.dropna(subset=['Antigen_AA'], inplace=True)

print('Removed observations (rows) which do not have standard aminoacids for antigen peptides \nObservations: {} \n'.format(peptide_detail_TCRb_2.shape[0]))

peptide_detail_TCRb_2['CDR3'] = g(peptide_detail_TCRb_2['CDR3'])
peptide_detail_TCRb_2['CDR3'].replace('', np.nan, inplace=True)
peptide_detail_TCRb_2.dropna(subset=['CDR3'], inplace=True)

print('Removed observations (rows) which do not have standard aminoacids for CDR3 \nObservations: {} \n'.format(peptide_detail_TCRb_2.shape[0]))

#%% TCRb

# Only keep CDR3 seqeunces which start with 'C' and end with 'F'

def proper_TCRb(x):
    """Returns only amino acids which start with 'C' and end with 'F'. Then keeps only the aa in between those letters.  

    Parameters
    ----------
    x : str
        CDR3 peptide sequence

    Returns
    -------
    x: str/ NaN
        Middle part of the string. If input starts with 'C' and ends with 'F'
        Otherwise None.
    """
    if x[0] == 'C' and x[-1] == 'F':
        x = x.replace(x[0],'')
        x = x.replace(x[-1],'')
        return x
    else:
        return np.NaN

f = np.vectorize(proper_TCRb, otypes=[pd.core.series.Series])
peptide_detail_TCRb_3 = peptide_detail_TCRb_2.copy()
peptide_detail_TCRb_3['CDR3'] = f(peptide_detail_TCRb_3['CDR3'])
peptide_detail_TCRb_3.dropna(subset=['CDR3'], inplace=True)

print('Only keep CDR3 AA seqeunces which start with C and end with F \
      \n Kepts the AAs between C and F. \nObservations: {} \n'.format(peptide_detail_TCRb_3.shape[0]))
#%%

# Check if there are any null values for the Antigen_AA column: those could be used as negatives for the CDR3 sequences: 

are_negatives = peptide_detail_TCRb_3['Antigen_AA'].isnull().values.any()
# Un-comment this line to check:
# print(are_negatives)
# if False, then there are no negatives

print('Are there any CDR3-antigen binding negatives? {}'.format(are_negatives))
print('Confirmation: \nNo NaN values so all are positives - peptide_detail_TCRb_3 is a dataset with only positives \n')

#%%

# Add HLA-A_1 and HLA_A_2 empty columns:
# Obs left: 53704 (not affected)
peptide_detail_TCRb_3['HLA-A_1'] = np.NaN
peptide_detail_TCRb_3['HLA-A_2'] = np.NaN


# subject_metadata: Give column 8 and 9 into different names (HLA-A-1 and HLA-A-2):
subject_metadata = subject_metadata.rename(columns={ subject_metadata.columns[8]: 'HLA-A-1' })
subject_metadata = subject_metadata.rename(columns={ subject_metadata.columns[9]: 'HLA-A-2' })

# keep columns of interest
subject_metadata = subject_metadata[['Experiment','HLA-A-1','HLA-A-2']]

#%%
# Mapping HLA values between peptide_detail_TCRb and subject_metadata
# NB: No. of observations remains unchanged for this dataset

#Convert dataframes to numpy
peptide_details_np = peptide_detail_TCRb_3.to_numpy()
subject_metadata_np = subject_metadata.to_numpy()

np_pep = peptide_details_np.copy()
row_peptide_details = 0

# Map HLA-A alleles presented in the patients (when the sampling was done): 
# based on the Experiment ID which matches with np_pep and subject_metadata_np
for exp_id in (np_pep[:,1]):
    exp_id1 = exp_id
    index_arr = (np.where(subject_metadata_np[:,0]==exp_id1))[0]
    row_metadata = index_arr[0] #access row index of metadata
    val_hla1 = subject_metadata_np[row_metadata,1] #Access value of HLA-A-1
    val_hla2 = subject_metadata_np[row_metadata,2] #Access value of HLA-A-2
    np_pep[row_peptide_details,3] = val_hla1 #update value for HLA-A-1 column at specified index
    np_pep[row_peptide_details,4] = val_hla2 #update value for HLA-A-2 column at specified index
    row_peptide_details +=1

print('Mapped HLA values between peptide_detail and subject_metadata \nObservations: {} \n'.format(np_pep.shape[0]))

#%%
# Pivotting data

np_pep_2 = np_pep.copy()

HLA_2nd_column = np.array([np_pep_2[:,-1]]).reshape(np_pep_2[:,-1].size,1)
np_pep_2_2 = np.append(np_pep_2[:,0:3],HLA_2nd_column, axis = 1)

np_pep_double =  np.append(np_pep_2[:,0:4], np_pep_2_2, axis = 0)

print('Pivotted HLA columns \nObservations double: {} \n'.format(np_pep_double.shape[0]))

#%%
# Remove NaN values (which only come from HLA values since there are no NaN values in np_pep_double (peptide_details_np))
# NB: No. of observations remains unchanged for this dataset
test_nan = np.frompyfunc(lambda i: i is np.nan,1,1)(np_pep_double).astype(bool)
nan_rows_bool= np.any(test_nan, axis=1) # test whole rows
np_pep_double = np_pep_double[~nan_rows_bool,:]


print('Removed observations with missing HLA values (NaN)  \nObservations: {} \n'.format(np_pep_double.shape[0]))

#%% Rename HLA values
np_pep_double = np_pep_double.astype(str)

def rename_HLA(x):
    """Modifies the HLA string

    Parameters
    ----------
    x : str
        HLA string to be modified

    Returns
    -------
    hla: str
        Modified HLA string
    """
    x = x.replace('*','')
    x = x.replace(':','')
    hla = x[0:6]
    return hla

vect_rename_HLA = np.vectorize(rename_HLA)
np_pep_double[:,3] = vect_rename_HLA(np_pep_double[:,3])

#%%
# Keep only observations with HLA-A02:01 values
np_pep_idx1 = np.where(np_pep_double[:,3]=='A0201')
np_pep_data1 = np.take(np_pep_double, np_pep_idx1, axis=0)[0,:,:]

print('Kept only observations with HLA-A02:01 values  \nObservations: {} \n'.format(np_pep_data1.shape[0]))

#%%
# Keep peptides (from antigen) which are only 9 AAs in length
length_checker = np.vectorize(len) 
np_pep_idx2 = np.where(length_checker(np_pep_data1[:,2]) == 9)
np_pep_data = np.take(np_pep_data1, np_pep_idx2, axis=0)[0,:,:]

print('Kept only observations with peptides of only 9 AAs in length (antigen column)  \nObservations: {} \n'.format(np_pep_data.shape[0]))

#%% Extract data for netMHCpan

df_pep_netMHCpan = pd.DataFrame(np_pep_data[:,2],  columns=['Antigen Peptide'])

# Remove peptide duplicates
df_pep_netMHCpan = df_pep_netMHCpan.drop_duplicates()
df_pep_netMHCpan= df_pep_netMHCpan.reset_index(drop=True)

# Uncomment this to export Excel file:

# Make Excel File of the data collected (saves in the processed_data_results folder) :
df_pep_netMHCpan2 = df_pep_netMHCpan.to_excel("./processet_data_results/Unique_peptides_with_HLA-A0201.xlsx",
            sheet_name='Unique peptides with HLA-A0201')

#%%
'''
Note: On this dataset netMHCpan has been used to prredict HLA-A0201 complex binding. 
https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1

Prediction has been exported in an xlsx file: NetMHCpan_HLAbinding_prediction.xlsx.
Prediction is stated in the NB column as 1/0
'''

df_peptide_binding = pd.read_excel(path + 'NetMHCpan_HLAbinding_prediction.xlsx')

df_peptide_binding.columns = df_peptide_binding.iloc[0]
df_peptide_binding = df_peptide_binding.drop(df_peptide_binding.index[0])

# Create a prediciton dictionary:
binding_score = dict(zip(df_peptide_binding['Peptide'], df_peptide_binding['NB']))

#%%
# Map HLA binding score values to the CDR3-peptide dataframe (df_pep_data)
df_pep_data = pd.DataFrame(np_pep_data, columns = ['CDR3','Experiment','Peptide_antigen','HLA-A'])
df_pep_data['binding_score'] = df_pep_data['Peptide_antigen'].map(binding_score)

print('Mapped HLA-A0201 binding score using netMHCpan  \nObservations: {} \n'.format(df_pep_data.shape[0]))

#%%
# Keep only positives
df_pep_data_positives = df_pep_data[df_pep_data.binding_score!=0]
df_pep_data_positives = df_pep_data_positives.reset_index(drop=True)

print('Kept only positive HLA-binding columns  \nObservations: {} \n'.format(df_pep_data_positives.shape[0]))

#%%
# Drop duplicates of CDR3- Antigen AA - HLA binding
df_pep_data_positives = df_pep_data_positives.drop_duplicates(subset=['CDR3','Peptide_antigen'])

print('Dropped duplicates of CDR3-Antigen peptide bindings  \nObservations: {} \n'.format(df_pep_data_positives.shape[0]))

#%%
# Drop CDR3 sequences which bind to multiple peptides. Obervations kept: none
# Note: since we eliminated CDR3-Peptide_antigen duplicates it is enough to remove the duplicates of CDR3 only
# since now they bind different peptides. 
df_pep_data_positives = df_pep_data_positives.drop_duplicates(subset=['CDR3'], keep=False)
df_pep_data_positives = df_pep_data_positives[['Experiment', 'CDR3', 'Peptide_antigen', 'HLA-A', 'binding_score']] #re-arranging the columns
df_pep_data_positives = df_pep_data_positives.reset_index(drop=True)

print('Dropped duplicates of CDR3 binding to multiple peptides  \nObservations: {} \n'.format(df_pep_data_positives.shape[0]))

#%%
# Export to Excel: df_pep_data_positives

df_pep_data3 = df_pep_data_positives.to_excel("./processet_data_results/peptide_dataset_wHLA_positives.xlsx",
            sheet_name='Positives - CDR3 and peptide ')

print("Exported df_pep_data_positives to Excel files:\n peptide_dataset_wHLA_positives.xlsx\n")

#%%

# CREATE PEPTIDE DATA SET WITH NEGATIVES

# Create a TCR dictionary - having CDR3 as key and peptide as value:
TCR_dict = dict(zip(df_pep_data_positives['CDR3'], df_pep_data_positives['Peptide_antigen']))
len(TCR_dict)
print("Created CDR3-peptide binding dictionary of positives \n")

# Shuffle the dictionary (to prevent a sequence of the same peptides at the end - else mismatching will not work):
random.seed(0)
TCR_dict_list = list(TCR_dict.items())
random.shuffle(TCR_dict_list)
TCR_dict = dict(TCR_dict_list)

# Create the pool of peptides - keeping doubles if peptide binds to multiple CDR3
peptide_counts = pd.DataFrame(df_pep_data_positives['Peptide_antigen'].value_counts())

unique_pep = peptide_counts.index.tolist()

peptide_counts_dict = dict(zip(unique_pep, peptide_counts['Peptide_antigen']))

# Un-comment this line to get an overview:
#print(max(peptide_counts_dict.items(), key=operator.itemgetter(1)))

def mismatch_peptide(dict_val):
    """Mismatches peptides using optimized dictonaries (optimized time).
    To be mapped on each key value of TCR_dict

    Parameters
    ----------
    dict_val : str
        Peptide input to be excluded from pep_counts_dict

    Returns
    -------
    pep_assigned
        Peptide with highest occurence value from peptide_counts_dict (excluding dict_val)
    """
    pep_excluded = dict_val
    peptide_counts_dict_excl = peptide_counts_dict.copy()
    peptide_counts_dict_excl.pop(pep_excluded)
    
    pep_assigned = max(peptide_counts_dict_excl.items(), key=operator.itemgetter(1))[0]

    if peptide_counts_dict[pep_assigned] >= 1:
        peptide_counts_dict[pep_assigned] = peptide_counts_dict[pep_assigned] - 1
    else: 
        peptide_counts_dict[pep_assigned] = peptide_counts_dict[pep_assigned]
    
    
    return pep_assigned

TCR_dict2 = dict((k, mismatch_peptide(v)) for k, v in TCR_dict.items())

shared_items = {k: TCR_dict[k] for k in TCR_dict if k in TCR_dict2 and TCR_dict[k] == TCR_dict2[k]}
  
print('Number of common element between the two dictionaries: {}\n'.format(len(shared_items)))

#%%
# Mismatching the TCR-Peptide bindings to create a dataset with negatives:
df_pep_data_negatives = df_pep_data_positives.copy()
df_pep_data_negatives['Peptide_antigen'] = df_pep_data_negatives['CDR3'].map(TCR_dict2)
df_pep_data_negatives['binding_score'] = 0

# Drop the duplicates, if any assigned:
df_pep_data_negatives = df_pep_data_negatives.drop_duplicates(subset=['CDR3', 'Peptide_antigen'], keep=False)

print('Created peptide dataset with negatives (peptide mismatches)\nObservations: {}\n'.format(df_pep_data_negatives.shape[0]))

print('Dataset with positives: {} observations\n'.format(df_pep_data_negatives.shape[0]), \
      'Dataset with negatives: {} observations\n'.format(df_pep_data_positives.shape[0]), \
      'Is their length equal? {}\n'.format(df_pep_data_negatives.shape[0]==df_pep_data_positives.shape[0]))

print('pos:neg-ratio is of 1:1\n')

#%%
# Converge datasets: df_pep_data_positives with df_pep_data_negatives

df_pep_data_pos_neg = pd.concat([df_pep_data_positives, df_pep_data_negatives], axis=0)
df_pep_data_pos_neg = df_pep_data_pos_neg.reset_index(drop=True)

print('Created peptide dataset with both positives and negatives (50/50 balanced set)  \nObservations: {} \n'.format(df_pep_data_pos_neg.shape[0]))

# Make Excel File of the data collected (saves in the processed_data_results folder) :
df_pep_data_pos_neg2 = df_pep_data_pos_neg.to_excel("./processet_data_results/peptide_dataset_wHLA_pos_neg.xlsx",
            sheet_name='CDR3 and Peptide binding ')

print('Exported into Excel file in the processed_data_results folder: peptide_dataset_wHLA_pos_neg.xlsx')






