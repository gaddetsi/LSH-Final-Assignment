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

# Minhashing
def minhash(A: csc_matrix, h: int, seed: int) -> np.ndarray:
 
    rng = np.random.default_rng(seed)
    n_movies, n_users = A.shape
    sigs = np.empty((h, n_users), dtype=np.int32)

    # Column index pointers and row indices of CSC
    colptr = A.indptr
    rows = A.indices  

    # Handle users with at least 1 rating 
    # For each i in 0..h-1 we make a permutation rank for rows and take min per column

    for i in range(h):
        perm = rng.permutation(n_movies).astype(np.int32)   # perm[old_row] -> new_row_pos

        # Map every stored nonzero to its permuted "rank"
        vals = perm[rows]

        # Per-column minimum in one vectorized sweep
        mins = np.minimum.reduceat(vals, colptr[:-1])
        sigs[i, :] = mins
        
    return sigs

# LSH
def lsh(signatures: np.ndarray, r: int, b: int, cutoff: int = 500):
   
    h, n_users = signatures.shape
    needed = r * b
    used = signatures[:needed, :]  # (r*b, n_users)

    for band in range(b):
        start, end = band * r, (band + 1) * r
        buckets = defaultdict(list)
        for u in range(n_users):
            key = tuple(used[start:end, u])  # band signature for user u
            buckets[key].append(u)

        for us in buckets.values():
            k = len(us)
            if k < 2 or k >= cutoff:
                continue
            us.sort()
            for i in range(k - 1):
                for j in range(i + 1, k):
                    yield us[i], us[j]


# Compute Jaccard
def jaccard_over_threshold(A: csc_matrix, u1: int, u2: int, thr: float = 0.5) -> bool:
   
    cptr = A.indptr
    r = A.indices
    a = r[cptr[u1]:cptr[u1+1]]
    b = r[cptr[u2]:cptr[u2+1]]

    # Two-pointer intersection count on sorted arrays
    i = j = inter = 0
    na, nb = a.size, b.size
    while i < na and j < nb:
        if a[i] == b[j]:
            inter += 1; i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    union = na + nb - inter
    if union == 0:
        return False
    return (inter / union) > thr

# Write Output
def write_pair_line(path: Path, u1: int, u2: int):
    with path.open("a") as f:
        f.write(f"{u1},{u2}\n")

# Main 
def main():
    seed = parse_seed()  
    os.makedirs(Path(OUT).parent, exist_ok=True)

    # 1) Load data
    A = load_user_movie_matrix(DATA)
    n_movies, n_users = A.shape
    print(f"Loaded matrix: {n_movies} movies x {n_users} users")

    # 2) Minhash via permutations
    sigs = minhash(A, h=H, seed=seed)  # shape (H, n_users)

    # 3) LSH (
    out_path = Path(OUT)
    if out_path.exists():
        out_path.unlink()

    seen = set()
    kept = 0
    for u1, u2 in lsh(sigs, r=R, b=B, cutoff=BIG_BUCKET_CUTOFF):
        if u1 > u2:
            u1, u2 = u2, u1
        pair = (u1, u2)
        if pair in seen:
            continue
        seen.add(pair)

        if jaccard_over_threshold(A, u1, u2, thr=THR):  # strict >
            write_pair_line(out_path, u1 + 1, u2 + 1)   # output 1-based IDs
            kept += 1

    print(f"Done. Wrote {kept} pairs to {out_path}")

if __name__ == "__main__":
    sys.exit(main())