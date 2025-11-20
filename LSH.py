# Import Neccesary Libraries
import os, sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

# Parameters
DATA = "user_movie_rating.npy"  
OUT  = "pairs_seed432.txt"
H    = 100
R    = 7        
B    = 14
THR  = 0.5
BIG_BUCKET_CUTOFF = 500

# Seed from command line
def parse_seed():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args().seed

# Load Data
def load_user_movie_matrix(npy_path: str) -> csc_matrix:
    
    p = Path(npy_path)

    arr = np.load(p, mmap_mode="r")  # (N,3) integers
    u = arr[:, 0].astype(np.int64)
    m = arr[:, 1].astype(np.int64)

    # Dataset is 1-based, convert consecutive - zero-base
    u0 = (u - u.min()).astype(np.int64)
    m0 = (m - m.min()).astype(np.int64)
    n_users = int(u0.max()) + 1
    n_movies = int(m0.max()) + 1
    data = np.ones_like(u0, dtype=np.uint8)
    A = coo_matrix((data, (m0, u0)), shape=(n_movies, n_users)).tocsc()

    return A


# Function to compute Minhash Signatures
def minhash_signatures(matrix, hash_num, random_seed):
    
    random_number=np.random.default_rng(random_seed) # Initializing random number generator with seed
    num_movies, num_users = matrix.shape # Getting number of movies and users from the matrix shape
    signature_matrix = np.zeros((hash_num, num_users), dtype=np.int32) # Initializing signature matrix with zeros
    rated_movies = matrix.indices 
    
    # Computing Minhash Signatures
    for i in range(hash_num): 

        permutation = random_number.permutation(num_movies) # Generating a random permutation of movie indices
        new_positions = permutation[rated_movies] # Mapping rated movies to their new positions in the permutation
        min_values = np.minimum.reduceat(new_positions, matrix.indptr[:-1]) # Computing minimum values for each user using reduceat
        signature_matrix[i] = min_values 
    
    return signature_matrix

