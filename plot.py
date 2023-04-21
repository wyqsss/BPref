import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

colors = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2], sns.color_palette()[4], sns.color_palette()[5], sns.color_palette()[3], sns.color_palette()[6], sns.color_palette()[7]]


def plot_muticurve(ori_path, color=None):
    sds = os.listdir(ori_path)
    suc_rates = []
    steps = None
    for s in sds:
        csv_path = os.path.join(ori_path, s, "train.csv")
        
        data = np.loadtxt(open(csv_path,"rb"),delimiter=",",skiprows=1, usecols=[3])
        suc_rates.append(data)
        if steps is None:
            steps = np.loadtxt(open(csv_path,"rb"),delimiter=",",skiprows=1, usecols=[1])

    suc_rates = np.asarray(suc_rates)
    mean_ = np.mean(suc_rates, axis=0)
    std_ = np.std(suc_rates, axis=0)
    top_ = mean_ + std_
    buttom_ = mean_ - std_
    plt.plot(steps, mean_, color=color)
    plt.fill_between(steps, top_, buttom_, alpha=0.3, color=color)
    
plot_muticurve("logs/PPO/metaworld_door-open-v2", color=colors[0])
plt.title("door-open-v2")
plt.xlabel("steps")
plt.ylabel("success rate")
plt.grid()
plt.savefig("figures/dooropen.png")


plt.close()





# kill -9 `ps -ef |grep Agent|awk '{print $2}' `