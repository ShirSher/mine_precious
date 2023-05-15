# %% [code] {"execution":{"iopub.status.busy":"2023-04-24T14:49:34.788566Z","iopub.execute_input":"2023-04-24T14:49:34.788973Z","iopub.status.idle":"2023-04-24T14:49:35.250826Z","shell.execute_reply.started":"2023-04-24T14:49:34.788937Z","shell.execute_reply":"2023-04-24T14:49:35.249084Z"}}
import sys

print(sys.version)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

x = np.random.normal(size=1000000)
x+=1
count, bins, patches = plt.hist(x,bins=100)
non_zero_mask = (count != 0)
count = count[non_zero_mask]

# %% [code] {"execution":{"iopub.status.busy":"2023-04-24T14:49:35.253491Z","iopub.execute_input":"2023-04-24T14:49:35.253906Z","iopub.status.idle":"2023-04-24T14:49:35.261863Z","shell.execute_reply.started":"2023-04-24T14:49:35.253866Z","shell.execute_reply":"2023-04-24T14:49:35.260428Z"}}
probs = count / sum(count)
h = -(probs * np.log2(probs)).sum()
print("Entropy:", h)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-24T14:49:35.263416Z","iopub.execute_input":"2023-04-24T14:49:35.263764Z","iopub.status.idle":"2023-04-24T14:49:35.275036Z","shell.execute_reply.started":"2023-04-24T14:49:35.263733Z","shell.execute_reply":"2023-04-24T14:49:35.273583Z"}}
#value,counts = np.unique(x, return_counts=True)
probs = count / len(x)
ent = -probs * np.log2(probs)
h = ent.sum()
print("Entropy:", h)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-24T14:49:35.276612Z","iopub.execute_input":"2023-04-24T14:49:35.276985Z","iopub.status.idle":"2023-04-24T14:49:35.288830Z","shell.execute_reply.started":"2023-04-24T14:49:35.276949Z","shell.execute_reply":"2023-04-24T14:49:35.287435Z"}}
from scipy.stats import entropy

# calculate the entropy of the probability distribution
h = entropy(count, base=2)

print("Entropy:", h)