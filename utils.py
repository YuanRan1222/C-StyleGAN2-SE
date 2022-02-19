import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return


def plot_gif(data, xlim, ylim, xticks_num, yticks_num, xylabel, legend, saved_path, time_interval=1, figsize=(15, 3), dpi=100):
    def animate(i):
        plt.clf()
        plt.rcParams["image.cmap"] = "gray"
        plt.plot(data[i].T)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.linspace(xlim[0], xlim[1], xticks_num))
        plt.yticks(np.linspace(ylim[0], ylim[1], yticks_num))
        plt.xlabel(xylabel[0])
        plt.ylabel(xylabel[1])
        plt.legend(legend, loc=1)
        plt.title("Random: {}".format(i + 1))
        plt.tight_layout()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=np.shape(data)[0], interval=time_interval, blit=False)
    ani.save(saved_path, writer="pillow", fps=60)
    plt.close()