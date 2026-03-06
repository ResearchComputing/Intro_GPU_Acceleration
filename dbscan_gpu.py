#create dataset with 100,000 points 
print(f" ")
print(f"Generating Dataset")
print(f"....")
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=70000, factor=.35, noise=.05)

#convert dataset to Pandas DataFrame
import pandas as pd
import cudf
X_df = pd.DataFrame({'fea%d'%i: X[:,i] for i in range(X.shape[1])})
X_gpu = cudf.DataFrame.from_pandas(X_df)

# start timing
print(f"Starting DBSCAN")
print(f"....")
import time
start_time = time.perf_counter()

#run GPU-accelerated DBSCAN
from cuml import DBSCAN
db = DBSCAN(max_mbytes_per_batch=1000, eps=0.6, min_samples=2)
y_db = db.fit_predict(X)

#stop timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time of DBSCAN: {elapsed_time:.4f} seconds")
