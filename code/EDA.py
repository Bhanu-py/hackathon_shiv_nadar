import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("E:\Hackathon\Machine_Hack\Shiv_Nadar_DAC")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/train.csv")
print(df.head())

print(df.info())

catcols = [col for col in df.columns if df[col].dtype == "O"]

fig, ax = plt.subplots(3, 2)
axe = ax.ravel()
for k, i in enumerate(catcols):
    print(i)
    print(df[i].value_counts())
    df[i].value_counts().plot(kind="bar", ax=axe[k]).set_title(i)
plt.show()

# fig, axs = plt.subplots(3, 2, sharey=True, tight_layout=True)
# axs[0].hist(df["Outlet_ID"],)
# axs[1].hist(df["Outlet_ID"],)
# plt.hist(df["Outlet_ID"])
# plt.show()
# print(df["Outlet_ID"])

