
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix

DATA = "user_movie_rating.npy"
PAIRS = "pairs_seed123.txt"
SAMPLE = 200

arr = np.load(DATA, allow_pickle=True)
u = arr[:,0].astype(np.int64); m = arr[:,1].astype(np.int64)
u0 = u - u.min(); m0 = m - m.min()
A = coo_matrix((np.ones_like(u0), (m0, u0)),
               shape=(m0.max()+1, u0.max()+1)).tocsc()

def jacc(u1,u2):
    cptr, idx = A.indptr, A.indices
    a = idx[cptr[u1]:cptr[u1+1]]; b = idx[cptr[u2]:cptr[u2+1]]
    i=j=inter=0
    while i < a.size and j < b.size:
        if a[i]==b[j]: inter+=1; i+=1; j+=1
        elif a[i]<b[j]: i+=1
        else: j+=1
    union = a.size + b.size - inter
    return inter/union if union else 0.0

pairs = [tuple(map(int, l.split(","))) for l in Path(PAIRS).read_text().splitlines()]
min_uid = int(u.min())  # map back to 0-based
pairs0 = [(p[0]-min_uid, p[1]-min_uid) for p in pairs[:min(SAMPLE, len(pairs))]]
vals = [jacc(a,b) for (a,b) in pairs0]
if vals:
    arrv = np.array(vals)
    print(f"checked {len(arrv)} pairs → min={arrv.min():.3f}, median={np.median(arrv):.3f}, max={arrv.max():.3f}")
    print('≤0.5 count:', int((arrv<=0.5).sum()))
else:
    print("no pairs to check")

