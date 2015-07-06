''' Functions and classes for doing Gaussian process models of proteins'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import math

######################################################################
# Here are the functions associated with Hamming kernels
######################################################################

def hamming_kernel (seq1, seq2, var_p=1):
    """ Returns the number of shared amino acids between two sequences"""
    # (RS- Iterates through each sequence, comparing pairs of amino acids, one fromeach sequence. 
    # zip() function zips together two or more lists, i.e. returns a list where its first element is a list of the first element of list 1 and list 2, etc)
    return sum ([1 if str(a) == str(b) else 0 for a,b in zip(seq1, seq2)])*var_p
    
def make_hamming (seqs):
    """ Returns a hamming matrix for two or more sequences of the same length

    Parameters:  
        seqs (DataFrame)
        # (RS- seqs is a DataFrame, i.e. a 2-D array or matrix)
    
    Returns: 
        DataFrame
    """
    # note: could probably do this more elegantly without transposing
    # (RS-Gives the total number of rows in seqs DataFrame, which represents the total number of sequences)
    n_seqs = len (seqs.index)
    #(RS-np.zeros function returns an n_seqs x n_seqs zero (empty) matrix where n_seqs is the number of sequences)
    hamming = np.zeros((n_seqs, n_seqs))
    # (RS- Iterate through a tuple where the 1st element is (0,1st sequence), the 2nd element is (1,2nd sequence), etc (n X 1) THRU ROWS)
    for n1,i in zip(range(n_seqs), seqs.index):
        # (RS- Iterate through a tuple where the 1st element is (0,1st sequence), the 2nd element is (1,2nd sequence), etc (1 X n) THRU COLS)
        for n2,j in zip(range (n_seqs), seqs.transpose().columns):
            # (RS- sets seq1 to the location number of the ith sequence (ROW) in the (n X 1) seqs DataFrame)
            seq1 = seqs.loc[i]
            # (RS- sets seq2 to the location number of the jth sequence (COL) in the transposed (1 X n) seqs DataFrame)
            seq2 = seqs.transpose()[j]
            # (RS- fills hamming matrix with the number of shared residues between the ith and jth sequence)
            hamming[n1,n2] = hamming_kernel (seq1, seq2)
    # (RS- sets hamming_df to be the above created n X n hamming matrix, where n dentoes the number of sequences in seqs)
    hamming_df = pd.DataFrame (hamming, index = seqs.index, columns = seqs.index)
    return hamming_df          
        
######################################################################
# Here are functions specific to the structure-based kernel
######################################################################
# (RS-WAYS TO FIND VAR_P AND VAR_N DESCRIBED IN ROMERO PAPER)
#(RS-This is not what we use to generate our NCR library, ignore this.)
def generate_library (n_blocks, n_parents):
    """Generates all the chimeras with n_blocks blocks and n_parents parents
    
    Parameters: 
        n_blocks (int): number of blocks
        n_parents (int): number of parents
        
    Returns: 
        list: all possible chimeras
    """
    if n_blocks > 1:
        this = ([i+str(n) for i in generate_library(n_blocks-1,n_parents) for n in range (n_parents)])
        return this
    return [str(i) for i in range (n_parents)]

#(RS-These "contacts" refer to residue pairs that are within a certain distance of each other (~4.5 angstroms) in the parent sequence(s); check to see if these 
# contacts are present in the chimera sequences, returns a list containing 1 if the contact is present or 0 if the contact is not present)
def contacts_X_row (seq, contact_terms, var_p):
    """ Determine whether the given sequence contains each of the given contacts
    
    Parameters: 
        seq (iterable): Amino acid sequence
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))
        var_p (float): underlying variance of Gaussian process  
          
    Returns: 
        list: 1 for contacts present, else 0
    """
    X_row = []
    for term in contact_terms:
        # (RS- If the protein sequence contains matching residues at the correct positions as the contacting residues in the contact_terms list, add 1 to the 
        # list, otherwise add 0, return the list)
        if seq[term[0][0]] == term[0][1] and seq[term[1][0]] == term[1][1]:
            X_row.append (1)
        else:
            X_row.append (0)
    return [var_p*x for x in X_row]

# (RS- Determines shared conatacts between two sequences)    
def structure_kernel (seq1, seq2, contact_terms, var_p=1):
    """ Determine the number of shared contacts between the two sequences
    
    Parameters: 
        seq1 (iterable): Amino acid sequence
        seq2 (iterable): Amino acid sequence
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))
        var_p (float): underlying variance of Gaussian process  
  
    Returns: 
        int: number of shared contacts
    """
    X1 = contacts_X_row (seq1, contact_terms)
    X2 = contacts_X_row (seq2, contact_terms)
    # Will take var_p into account when taking the dot product of the binary vector (Romero))
    return sum ([1 if a == b else 0 for a,b in zip(X1, X2)])*var_p

# (RS- Creates a list containing every list of contacting terms for each sequence in seqs (Binary vector))    
def make_contacts_X (seqs, contact_terms,var_p=1):
    """ Makes a list with the result of contacts_X_row for each sequence in seqs"""
    
    X = []
    for i in seqs.index:
        X.append(contacts_X_row(seqs.loc[i],contact_terms,var_p))
    return X

# (RS- creates the structure-based covariance matrix, each element ij in the matrix will correspond to the dot product of the binary vectors 
# multiplied by var_p, of two sequences, including a sequence and itself)        
def make_structure_matrix (seqs, contact_terms,var_p=1):
    """ Makes the structure-based covariance matrix
    
    Parameters: 
        seqs (DataFrame): amino acid sequences
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))    
        var_p (float): underlying variance of Gaussian process  
    
    Returns: 
        Dataframe: structure-based covariance matrix
    """
    X = np.matrix(make_contacts_X (seqs, contact_terms,var_p))
    # (RS- Index=row)
    return pd.DataFrame(np.einsum('ij,jk->ik', X, X.T), index=seqs.index, columns=seqs.index)

# (RS-Returns a list of all of the contacts in the form ((residue position number, amino acid number),(residue position number,amino acid number)))
def contacting_terms (sample_space, contacts):
    """ Lists the possible contacts
    
    Parameters: 
        sample_space (iterable): Each element in sample_space contains the possible 
           amino acids at that position
        contacts (iterable): Each element in contacts pairs two positions that 
           are considered to be in contact
    
    Returns: 
        list: Each item in the list is a contact in the form ((pos1,aa1),(pos2,aa2))
    """
    contact_terms = []
    for contact in contacts:
        first_pos = contact[0]
        second_pos = contact[1]
        first_possibilities = set(sample_space[first_pos])
        second_possibilities = set(sample_space[second_pos])
        for aa1 in first_possibilities:
            for aa2 in second_possibilities:
                contact_terms.append(((first_pos,aa1),(second_pos,aa2)))
    return contact_terms


######################################################################
# Here are tools that are generally useful
######################################################################

def K_factor (K, var_p, var_n):
    """ Calculates the inverted matrix used to predict mean and variance.

    This is [var_p*K+var_n^2*I]^-1. 

    Parameters: 
        K (numpy.matrix): 
        var_p (float): The hyperparameter to use in in the kernel function
        var_n (float): Signal noise variance
          
    Returns: 
        (np.matrix): The result is returned as a matrix    
    """
    return np.linalg.inv(var_p*K+var_n*np.identity(len(K)))

def plot_predictions (real_Ys, predicted_Ys,stds=None,file_name=None,title='',label='', line=True):
    if stds is None:
        plt.plot (real_Ys, predicted_Ys, 'g.')
    else:
        plt.errorbar (real_Ys, predicted_Ys, yerr = [stds, stds], fmt = 'g.')
    small = min(set(real_Ys) | set(predicted_Ys))*1.1
    large = max(set(real_Ys) | set(predicted_Ys))*1.1
    if line:
        plt.plot ([small, large], [small, large], 'b--')
    plt.xlabel ('Actual ' + label)
    plt.ylabel ('Predicted ' + label)
    plt.title (title)
    plt.text(small*.9, large*.7, 'R = %.3f' %np.corrcoef(real_Ys, predicted_Ys)[0,1])
    if not file_name is None:
        plt.savefig (file_name)
        
def plot_LOO(Xs, Ys, args=[], kernel='Hamming',save_as=None, lab=''):
    std = []
    predicted_Ys = []
    for i in Xs.index:
        train_inds = list(set(Xs.index) - set(i))
        train_Xs = Xs.loc[train_inds]
        train_Ys = Ys.loc[train_inds]
        verify = Xs.loc[[i]]
        if kernel == 'Hamming':
            predicted = HammingModel(train_Xs,train_Ys,guesses=[.001,.250]).predict(verify)
        [(E,v)] = predicted
        std.append(math.pow(v,0.5))
        predicted_Ys.append (E)
    plot_predictions (Ys.tolist(), predicted_Ys, stds=std, label=lab, file_name=save_as)

######################################################################
# Now we start class definitions
######################################################################  
      
class GPModel(object):
    """A Gaussian process model for proteins. 

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (DataFrame): The outputs for the training set
        K (DataFrame): Covariance matrix
        inv_K_factor (np.matrix): The inverted matrix used in making predictions
        ML (float): The negative log marginal likelihood
    """
    
    #__metaclass__ = ABCMeta
    
    def __init__ (self, sequences, outputs, hyper=None, inv_factor=None):
        if sequences.index != outputs.index:
            print ('Warning: mismatch between training sequences and outputs')
        self.X_seqs = sequences
        self.Y = outputs
        self.hyper = hyper
        self.inv_factor = inv_factor

    
            
    def predict (self, k, k_star):
        """ Predicts the mean and variance of the output for each of new_seqs
        
        Uses Equations 2.23 and 2.24 of RW
        
        Parameters: 
            k (np.matrix): k in equations 2.23 and 2.24
            k_star (float): k* in equation 2.24
            
        Returns: 
            res (tuple): (E,v) as floats
        """
        E = k*self.inv_factor*np.matrix(self.Y).T
        v = k_star - k*self.inv_factor*k.T
        return (E.item(),v.item())
    
    def log_ML (self,factor):
        """ Returns the negative log marginal likelihood.  
        
        Parameters: 
            factor (np.matrix): the inverse of [var_p*K+var_n*I]
    
        Uses RW Equation 5.8
        """
        Y_mat = np.matrix(self.Y)
        inv_factor = np.linalg.inv(factor)
        """first = 0.5*Y_mat*inv_factor*Y_mat.T
        second = math.exp(0.5)*np.linalg.det(factor).item()
        third = math.exp(len(Y_mat)/2.)*2*math.pi
        try:
            return first + math.log(second*third)
        except:
            exit (str(second))"""
        L = (0.5*Y_mat*inv_factor*Y_mat.T + 0.5*math.log(np.linalg.det(factor)) + len(Y_mat)/2*math.log(2*math.pi)).item()
        return L
        
#(RS- GP model for proteins using a structure-based kernel)
class StructureModel(GPModel):
    """A Gaussian process model for proteins with a structure-based kernel

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (DataFrame): The outputs for the training set
        var_p (float): the hyperparameter to use in in the kernel function
        var_n (float): signal variance        
        inv_factor (np.matrix): The inverted matrix used in making predictions
        K (DataFrame): Covariance matrix
        factor (np.matrix): [var_p*K+var_n*I]
        contacts (iterable): Each element in contacts pairs two positions that 
               are considered to be in contact  
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))     
        sample_space (iterable): Each element in sample_space contains the possible 
           amino acids at that position
    """
    def __init__ (self, sequences, outputs, contacts, sample_space, guesses=[0.001,.250]):
        self.X_seqs = sequences
        self.Y = outputs
        self.contacts = contacts
        self.sample_space = sample_space
        self.contact_terms = contacting_terms(self.sample_space, self.contacts)
        self.K = structure_kernel (self.X_seqs, self.contact_terms)
        minimize_res = minimize(self.log_ML,(guesses), bounds=[(1e-5,None),(1e-5,None)])
        self.var_n,self.var_p = minimize_res['x']
        self.ML = minimize_res['fun']
        self.factor = self.var_p*np.matrix(self.K) + self.var_n*np.identity(len(self.K))
        self.inv_factor = np.linalg.inv(self.factor)

    def log_ML (self, variances):
        """ Returns the negative log marginal likelihood.  
    
        Uses RW Equation 5.8
    
        Parameters: 
            variances (iterable): var_n and var_p

        Returns: 
            L (float): the negative log marginal likelihood
        """
        var_n,var_p = variances
        K_mat = np.matrix (self.K)
        return super(StructureModel,self).log_ML(K_mat*var_p+np.identity(len(K_mat))*var_n)  
        
    def predict (self, new_seqs):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs
    
        Uses Equations 2.23 and 2.24 of RW and a structure-based kernel
        
        Parameters: 
            new_seqs (DataFrame): sequences to predict
            
         Returns: 
            predictions (list): (E,v) as floats
        """
        predictions = []
        for ns in [new_seqs.loc[i] for i in new_seqs.index]:
            k = np.matrix([structure_kernel(ns,seq1,self.var_p) for seq1 in [self.X_seqs.loc[i] for i in self.X_seqs.index]])
            k_star = structure_kernel(ns,ns,self.var_p)
            predictions.append(super(StructureModel,self).predict(k, k_star))
        return predictions  
         
#(RS- GP model for for proteins using a Hamming distance based kernel)
class HammingModel(GPModel):
    """A Gaussian process model for proteins with a Hamming kernel

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (DataFrame): The outputs for the training set
        var_p (float): the hyperparameter to use in in the kernel function
        var_n (float): signal variance        
        inv_factor (np.matrix): The inverted matrix used in making predictions
        K (DataFrame): Covariance matrix
        factor (np.matrix): [var_p*K+var_n*I]
    """
    
    def __init__ (self, sequences, outputs, guesses=[0.001,.250]):
        self.X_seqs = sequences
        self.Y = outputs
        self.K = make_hamming (self.X_seqs)
        minimize_res = minimize(self.log_ML,(guesses), bounds=[(1e-5,None),(1e-5,None)])
        self.var_n,self.var_p = minimize_res['x']
        self.ML = minimize_res['fun']
        self.factor = self.var_p*np.matrix(self.K) + self.var_n*np.identity(len(self.K))
        self.inv_factor = np.linalg.inv(self.factor)
        #self.inv_factor = K_factor (self.K, self.var_p, self.var_n)
        
    def log_ML (self, variances):
        """ Returns the negative log marginal likelihood.  
    
        Uses RW Equation 5.8
    
        Parameters: 
            variances (iterable): var_n and var_p

        Returns: 
            L (float): the negative log marginal likelihood
        """
        var_n,var_p = variances
        K_mat = np.matrix (self.K)
        return super(HammingModel,self).log_ML(K_mat*var_p+np.identity(len(K_mat))*var_n)
            
    def predict (self, new_seqs):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs
    
        Uses Equations 2.23 and 2.24 of RW and a hamming kernel
        
        Parameters: 
            new_seqs (DataFrame): sequences to predict
            
         Returns: 
            predictions (list): (E,v) as floats
        """
        predictions = []
        for ns in [new_seqs.loc[i] for i in new_seqs.index]:
            k = np.matrix([hamming_kernel(ns,seq1,self.var_p) for seq1 in [self.X_seqs.loc[i] for i in self.X_seqs.index]])
            k_star = hamming_kernel(ns,ns,self.var_p)
            predictions.append(super(HammingModel,self).predict(k, k_star))
        return predictions 
        