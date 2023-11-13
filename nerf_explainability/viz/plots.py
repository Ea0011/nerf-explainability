import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation


def plot_periodic_function(x_min=-1, x_max=1, max_log_band=10, fn=np.cos, ax=None, file_postfix=0):
    periodic_fn = lambda x: fn(x * (max_log_band - 1))
    xx = np.linspace(x_min, x_max, endpoint=True, num=1000)
    yy = periodic_fn(xx)
    
    if ax is not None:
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.plot(xx, yy, label=f"L = {max_log_band - 1}", lw=2)
        ax.legend(loc="upper right", fontsize="large")
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.plot(xx, yy, label=f"L = {max_log_band - 1}", linewidth=2)
        plt.legend(loc="upper right", fontsize="large")
        plt.savefig(f"./band_{file_postfix}.png", dpi=100)