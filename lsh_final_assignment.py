import numpy as np
import sys
from scipy.sparse import csr_matrix, csc_matrix
from collections import defaultdict
from datetime import datetime
import time

#command line argument parsing
def parse_arguments():
    if len(sys.argv) != 2:
        print("Usage: python3 lsh_final_assignment.py <random_seed>")
        sys.exit(1)
    
    try:
        return int(sys.argv[1])
    except ValueError:
        print("Error: Random seed must be an integer")
        sys.exit(1)


#load data from npy file
def load_data(filepath="user_movie_rating.npy"):
    print("Loading data...")
    
    # Load the data
    data = np.load(filepath)
    print(f"Data shape: {data.shape}")
    
    # Extracting columns
    user_ids = data[:, 0]
    movie_ids = data[:, 1]
    
    # Get dimensions
    n_users = int(user_ids.max())
    n_movies = int(movie_ids.max())
    print(f"Users: {n_users}, Movies: {n_movies}")
    
    # Adjust indices to 0-based 
    row_indices = movie_ids - 1
    col_indices = user_ids - 1
    values = np.ones(len(user_ids), dtype=np.int8)
    
    # Create CSC matrix 
    sparse_matrix = csc_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_movies, n_users),
        dtype=np.int8
    )
    
    print(f"Sparse matrix created: {sparse_matrix.shape}")
    print(f"Non-zero elements: {sparse_matrix.nnz:,}")
    
    # Create dictionary of user - set of movies 
    print("Building user-movie sets for verification...")
    user_movies_dict = defaultdict(set)
    for user_id, movie_id in zip(user_ids, movie_ids):
        user_movies_dict[user_id].add(movie_id)
    
    print(f"Data loading complete.\n")
    
    return sparse_matrix, n_movies, n_users, user_movies_dict

#jaccard similarity calculation
def calculate_jaccard_similarity(set1, set2):
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def minhash_signatures(sparse_matrix, n_users, signature_length=100):

    print(f"Computing MinHash signatures (h={signature_length})...")
    n_movies = sparse_matrix.shape[0]
    signature_matrix = np.full((signature_length, n_users), n_movies, dtype=np.int32)
    
    # Convert to CSR for efficient row operations
    sparse_csr = sparse_matrix.tocsr()
    
    for i in range(signature_length):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{signature_length} permutations")
        
        # Generate random permutation of movie indices
        permutation = np.random.permutation(n_movies)
        
        # Apply permutation: reorder rows according to permutation
        permuted_matrix = sparse_csr[permutation, :]
        
        # For each column (user), find the index of the first non-zero entry
        # Convert back to CSC for efficient column access
        permuted_csc = permuted_matrix.tocsc()
        
        for user_idx in range(n_users):
            col_data = permuted_csc.indices[permuted_csc.indptr[user_idx]:permuted_csc.indptr[user_idx+1]]
            if len(col_data) > 0:
                signature_matrix[i, user_idx] = col_data[0]
    
    print(f"MinHash signatures complete.\n")

    return signature_matrix

#banding function and candidate pair generation
def lsh_banding(signature_matrix, n_bands, rows_per_band, user_movies_dict, threshold=0.5, output_file='result.txt'):

    print(f"LSH Banding (b={n_bands}, r={rows_per_band})...")
    
    n_users = signature_matrix.shape[1]
    candidate_pairs = set()
    
    # Process each band
    for band_idx in range(n_bands):
        if (band_idx + 1) % 5 == 0:
            print(f"  Processing band {band_idx+1}/{n_bands}")
        
        # Extract rows for this band
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        band_signatures = signature_matrix[start_row:end_row, :]
        
        # Hash users to buckets based on their band signature
        buckets = defaultdict(list)
        for user_idx in range(n_users):
            # Create a tuple of the signature for this band
            band_sig = tuple(band_signatures[:, user_idx])
            buckets[band_sig].append(user_idx)
        
        # Collect candidate pairs from buckets with multiple users
        for bucket_users in buckets.values():
            if len(bucket_users) > 1:
                # Skip very large buckets 
                if len(bucket_users) > 100:
                    continue
                
                # Generate all pairs in this bucket
                for i in range(len(bucket_users)):
                    for j in range(i + 1, len(bucket_users)):
                        user1_idx = bucket_users[i]
                        user2_idx = bucket_users[j]
                        # Store as tuple with smaller index first
                        pair = (min(user1_idx, user2_idx), max(user1_idx, user2_idx))
                        candidate_pairs.add(pair)
    
    print(f"  Found {len(candidate_pairs)} candidate pairs")
    
    # Verify candidates and write to file
    print(f"Verifying candidates (threshold={threshold})...")
    similar_pairs_count = 0
    
    # Open file in write mode to clear previous contents
    with open(output_file, 'w') as f:
        pass
    
    verified_count = 0
    for user1_idx, user2_idx in candidate_pairs:
        verified_count += 1
        if verified_count % 10000 == 0:
            print(f"  Verified {verified_count}/{len(candidate_pairs)} candidates, found {similar_pairs_count} similar pairs")
        
        # Convert from 0-based index to 1-based user_id
        user1_id = user1_idx + 1
        user2_id = user2_idx + 1
        
        # Get movie sets for both users
        movies1 = user_movies_dict[user1_id]
        movies2 = user_movies_dict[user2_id]
        
        # Calculate exact Jaccard similarity
        jaccard_sim = calculate_jaccard_similarity(movies1, movies2)
        
        # If similarity exceeds threshold, write to file
        if jaccard_sim >= threshold:
            similar_pairs_count += 1
            # Write pair to file (with file open/close for safety)
            with open(output_file, 'a') as f:
                f.write(f"{user1_id},{user2_id}\n")
    
    print(f"\nVerification complete!")
    print(f"Total similar pairs found: {similar_pairs_count}")
    print(f"Results written to: {output_file}\n")
    
    return similar_pairs_count


if __name__ == "__main__":
    start_time = time.time()

    print("Starting....\n")

    random_seed = parse_arguments()
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}\n")

    sparse_matrix, n_movies, n_users, user_movies_dict = load_data("user_movie_rating.npy")

    signature_length = 100  # H
    signature_matrix = minhash_signatures(sparse_matrix, n_users, signature_length)

    n_bands = 20            # B 
    rows_per_band = 5       # R
    threshold = 0.5         # THR

    similar_pairs_count = lsh_banding(
        signature_matrix,
        n_bands,
        rows_per_band,
        user_movies_dict,
        threshold,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print("-"*70)
    print("LSH RUN COMPLETED")
    print(f"Total similar pairs (Jaccard > {threshold}): {similar_pairs_count}")
    print(f"Total runtime: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print("Results saved to: result.txt")
    print("-"*70)





