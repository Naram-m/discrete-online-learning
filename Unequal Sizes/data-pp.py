import pandas as pd
import numpy as np
from datetime import datetime

base_date = datetime(2018, 11, 21)
data = pd.read_csv("./ml-25m/ratings.csv", usecols=["movieId", "timestamp"])
data["timestamp"] = data.timestamp.apply(lambda x: datetime.fromtimestamp(x))
data = data.loc[data["timestamp"] >= base_date]

id, counts = np.unique(data["movieId"].values, return_counts=True)
frequencies = np.asarray((id, counts)).T
clipped = frequencies[frequencies[:, 1] >= 8] # at least two requests
print("Total # of files: {}".format(len(clipped)))

data = data.loc[data['movieId'].isin(clipped[:, 0])]
data = data.sort_values(by='timestamp', ascending=True)
consec_id = -1
for original_id in clipped[:, 0]:
    consec_id += 1
    if consec_id % 1000 == 0:
        print("#", end=" ")
    data["movieId"] = data["movieId"].replace(original_id, consec_id)
print()
'''At this point, you know that # of files is 9k. However, not all of them might appear in 
the trace. the trace is any 5k steps (or maybe make it 10k)
'''
# pick a 5k interval (not all 9k files will appear)
################## Testing ######################
# chunk = data.tail(20000)
chunk = data.head(10000)

# chunk = data.iloc[start : end]
# the below is to know how many files, out of the 9k, is present in the selected 10k time steps.
id2, counts2 = np.unique(chunk["movieId"].values, return_counts=True)
print(len(id2))

sizes = np.random.randint(1, 11, 10379)
with open('./ml_trace_10k_mini.npy', 'wb') as f:
    np.save(f, chunk.movieId.values)

with open('./weights.npy', 'wb') as f:
    np.save(f, sizes)

