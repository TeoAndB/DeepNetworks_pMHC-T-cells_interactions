# -*- coding: utf-8 -*-
"""
Helper Functions for Peptide Encoding according to the BLOSUM Matrix
"""
import numpy as np
import itertools as it 

#%%

def give_blosum_dict(blosum_filename):
    '''
    returns BLOSUM dictionary from specified blosum txt file
    parameters:
        - blosum_filename: txt file containing the BLOSUM matrix
    returns:
        - blosum_dict : BLOSUM dictionary having pair of AA and their correspondent value
    '''
    with open(blosum_filename, 'r') as f:
        lines = f.readlines()
    
    lines = [line for line in lines if not line.startswith('#')]
    header = lines[0].strip().split()
    header = header[:20]
    header.append('X')
    data = [[int(x) for x in line.strip().split()[1:]] for line in lines[1:]]
    data = np.asarray(data)[:21,:21]
    pad = np.zeros((21))
    data[-1,:] = pad
    data[:,-1] = pad.T
    blosum_dict = dict(zip(it.product(header, header), np.array(data).flatten()))
    
    return blosum_dict

#%%

def match_aa_code(k1, blosum_dict):
    '''
    returns list of matching codes for given aa
    parameters:
        - k1: amino acid
        - blosum_dict: BLOSUM matrix converted to dictionary
    returns:
        - encode_list : list containing the values in relation with the BLOSUM matrix
    '''
    aa_str = 'A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V'
    aa_list = aa_str.split(' ')
    aa_list = list(filter(None, aa_list))
    encode_list = []
    for k2 in aa_list:
        # encode.append(blosum_dict.get((k1, k2), blosum_dict[k2, k1]))
        
        if ((k1,k2) in blosum_dict):
            encode_list.append(blosum_dict.get((k1,k2)))
        elif ((k2,k1) in list(blosum_dict)):
            encode_list.append(blosum_dict.get((k2,k1)))
        else:
            print('Unknown amino acid in peptides: {}'.format(k1))
    return encode_list



def encoding_per_tcr(tcr, n_max, blosum_dict):
    '''
    blosum encoding of TCRb - with padding
    parameters:
        - tcr: TCR sequence
        - n_max: maximum length of a TCRb in the data set
        - blosum_dict: BLOSUM matrix converted to dictionary
    returns:
        - tcr_encoded : list containing the padded, encoded TCRb
    '''
    tcr_encoded = []
    tcr_aa = list(tcr) 
    rest_aa = ['X'] * int((n_max - len(tcr))) #padding added
    tcr_aa_list = tcr_aa + rest_aa
    
    for t in tcr_aa_list:
        l = match_aa_code(t, blosum_dict)
        tcr_encoded.append(l)
    
    return tcr_encoded


def encoding_per_peptide(pep, blosum_dict):
    '''
    blosum encoding of peptide - without padding
    parameters:
        - pep: peptide (antigen) sequence
        - blosum_dict: BLOSUM matrix converted to dictionary
    returns:
        - pep_encoded : list containing the padded, encoded peptide
    '''
    pep_aa_list= list(pep)
    
    pep_encoded = []
    for p in pep_aa_list:
        l = match_aa_code(p, blosum_dict)
        pep_encoded.append(l)
    
    return pep_encoded

#%% 
def matrix_encoding_per_TCR_list(tcr_list, n_max, blosum_dict):
    '''
    BLOSUM encoding for a list of TCRb
    parameters:
        - tcr_list: list of TCRb sequences
        - n_max: maximum length of a TCRb in the data set
        - blosum_dict: BLOSUM matrix converted to dictionary
    returns:
        - encoded_tcr_matrix : 3D array containing padded, encoded TCRb's
    '''
    encoded_tcr_list = []
    for tcr in tcr_list:
        tcr_m = encoding_per_tcr(tcr, n_max, blosum_dict)
        encoded_tcr_list.append(tcr_m)
    encoded_tcr_matrix = np.asarray(encoded_tcr_list, dtype=float)
    
    return encoded_tcr_matrix

def matrix_encoding_per_peptide_list(pep_list, blosum_dict):
    '''
    BLOSUM encoding for a list of peptides (antigens)
    parameters:
        - pep_list: list of peptides (antigens)
        - blosum_dict: BLOSUM matrix converted to dictionary
    returns:
        - encoded_tcr_matrix : 3D array containing padded, encoded peptides
    '''
    encoded_pep_list = []
    for pep in pep_list:
        pep_m = encoding_per_peptide(pep, blosum_dict)
        encoded_pep_list.append(pep_m)
    encoded_pep_matrix = np.asarray(encoded_pep_list, dtype=float)
    
    return encoded_pep_matrix
