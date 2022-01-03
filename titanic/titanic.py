# %%

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('titanic.csv')

data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})

# %%

fig, axs = plt.subplots(ncols=5, figsize=(30,5))
sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])

# %%
survival_sex = data[["sex", "survived"]].groupby("sex")
fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.histplot(data = survival_sex.get_group("male"), x="survived", ax=axs[0])
sns.histplot(data = survival_sex.get_group("female"), x="survived", ax=axs[1])

# %%
data.replace({'male': 1, 'female': 0}, inplace=True)
data.corr().abs()[["survived"]]

# %%
data["relatives"] = data.apply(lambda row: int(row["sibsp"] + row["parch"] > 0), axis=1)
data.corr().abs()[["survived"]]

# %%
data = data[['sex', 'pclass','age','relatives','fare','survived']].dropna()
data.corr().abs()[["survived"]]

features = data[['sex','pclass','age','relatives','fare']]
classes = data[['survived']].reset_index()

# %%
x_train, x_test, y_train, y_test = train_test_split(features, data.survived, test_size=0.2, random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

features_normalized = sc.fit_transform(features)

# %%
from sklearn.decomposition import PCA

pca = PCA()
components = pca.fit_transform(features_normalized)
df_components = pd.concat([pd.DataFrame(components), classes], axis=1)

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

survived = components[classes["survived"]==1]
not_survived = components[classes["survived"]==0]

ax.scatter(survived[:, 0], survived[:, 1], survived[:, 2])
ax.scatter(not_survived[:, 0], not_survived[:, 1], not_survived[:, 2])

# fig = plt.figure()
# ax2 = fig.add_subplot(111, projection="3d")
# ax2.scatter(components[:, 0], components[:, 1], components[:, 2])

plt.show()

# %%
import plotly.express as px

fig=px.scatter_3d(df_components, x=0, y=1, z=2, color="survived")
fig.show()

