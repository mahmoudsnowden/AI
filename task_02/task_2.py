from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMs


dataset = pd.read_csv('Wuzzuf_Jobs.csv')
dataset['YearsNum'] = pd.factorize(dataset['YearsExp'])[0]

# _________________________
X = dataset.iloc[:, [0, 1]].values
X = pd.DataFrame(X)
X = pd.get_dummies(X, columns=[0, 1])

wcss = []
for i in range(1, 20):
    kms = KMs(numClusters=i, init='k-means++', random_state=42)
    kms.fit(X)
    wcss.append(kms.inertia_)

plt.plot(range(1, 21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

knee1 = KneeLocator(range(1, 20), wcss, curve="convex", direction="decreasing")
print("elbow at : ", end="")
print(knee1.elbow)
kms = KMs(n_clusters=2, init='k-means++', random_state=42)
y_kms = kms.fit_predict(X)
y_kms == 0
