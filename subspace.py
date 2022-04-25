"""
Created on Mon Nov 18 15:56:20 2019

@author: Erica
"""

import numpy as np
import torch
import torch.nn.functional as F

def subspace_bases(X, num_vectors, contratio = 1.0, return_eigvals = False):
    """
    Return subspace basis using PCA
    Parameters
    ----------
    X:              Batch with sets of features. 
                    Tensor shape (batch_size, feature_dim, max_num_vectors)
    num_vectors:    Number of vectors in each set.
                    Tensor shape (batch_size)
    contRatio:      if int type, defines the dimensions of the subspaces.
                    if float type, defines the cummulative sum of variances that the basis
                    vectors of the subspaces should account for.
                    Scalar.
    return_eigvals: if True, this function also returns eigenvalues.
                    Bool.
    Returns
    -------
    V:          Basis of the subspaces.
                Tensor shape (n_batch, feature_dim, max_dim)
    w:          Eigenvalues
                Tensor shape (n_batch, max_dim)
    dims:       Dimensions of each subspace in the batch
                Tensor shape (n_batch)
    """

    n = X.shape[0]

    # 1) SVD:
    V, s, _ = torch.linalg.svd(X, full_matrices = False)
    w = s ** 2
    # 2) Define subspace dimensions
    if type(contratio) == int: # Actual subspace dimensions
        dims = contratio * torch.ones(n, dtype = torch.uint8)
        # Check if dim is larger than the number of vectors in each set
        mask = dims >= num_vectors
        indices = mask.nonzero()
        # If it is larger, make dim be num_vector
        if len(indices) > 0:
            #import pdb; pdb.set_trace()
            dims[indices] = num_vectors[indices]
    elif type(contratio) == float: # Cumulative ratio
        # Get cumulative variance
        cumvar = torch.cumsum(w, dim=1) / torch.sum(w, dim=1).reshape(n,1)
        mask = cumvar <= contratio
        dims = torch.cat([min((mask[i,:]==False).nonzero())+1 for i in range(n)])
        print(dims)

    # 3) Trim to max dim
    max_dim = int(max(dims))
    if V.shape[2] > max_dim:
        V = V.narrow(2, 1, max_dim)
        w = w.narrow(1, 1, max_dim)

    # 4) Zero out extra garbage
    for i in range(n):
        ind = dims[i]
        V[i,:,ind:] = 0
        w[i, ind:] = 0

    if return_eigvals:
        return V, w, dims
    else:
        return V, dims

def mean_square_singular_values(X, dims):
    """
    calculate mean square of singular values of X
    Parameters:
    -----------
    X : Tensor, shape: (n_sets, n_max_subdim1, n_max_subdim2)
    dims : Tensor, shape (n_sets, 2)
    Returns:
    --------
    c: mean square of singular values
    """

    # Frobenius norm is equivalent to square root of
    # sum of square singular values
    minimum, _ = torch.min(dims,1)
    mssv = torch.div(torch.mul(X, X).sum([1,2]),minimum.cuda())
    return 

def find_closest_subspace(ref_subspaces, ref_dims, input_subspaces, input_dims):
    
    [n_ref, featdim, max_ref_subdim] = ref_subspaces.shape
    
    closest = []

    for input in input_subspaces:

        input_subspace_expand = input.repeat(n_ref, 1, 1)

        dims = torch.zeros(n_ref,2)
        dims[:,0] = ref_dims
        dims[:,1] = input_dim

        # Get similarities
        ref_subspaces = torch.transpose(ref_subspaces, 1, 2)
        dot = torch.bmm(ref_subspaces, input_subspace_expand)
                
        sims = mean_square_singular_values(dot, dims=dims)

        # Find closest
        closest.append(torch.argmax(sims))

    return torch.cat(closest)
