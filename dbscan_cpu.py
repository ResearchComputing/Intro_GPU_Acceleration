
#create dataset with 100,000 points 
print(f" ")
print(f"Generating Dataset")
print(f"....")
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=70000, factor=.35, noise=.05)

# start timing
print(f"Starting DBSCAN")
print(f"....")
import time
start_time = time.perf_counter()

#run DBSCAN clustering algorithm
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.6, min_samples=2)
y_db = db.fit_predict(X)

#stop timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time of DBSCAN: {elapsed_time:.4f} seconds")

