import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ld_label(length, data):
    """
    Set up sfs caption, label and title.

    Parameter
    ---------
    length: length of the SFS
    title: title of the plot
    """
    # Caption
    plt.legend(loc="upper right", fontsize="x-large")
    # Label axis
    plt.xlabel("sequence", fontsize="xx-large")
    plt.ylabel("linckage desequilibrium", fontsize="xx-large")
    # Title + show
    title = (
        "LD  pour différents scénarios avec Ne={}, mu={}, taux de recombinaison={}"
        " & L={:.1e}"
    ).format(data[2]['Ne'], data[2]['mu'], data[2]['ro'], data[2]['length'])
    plt.title(title, fontsize="xx-large", fontweight='bold', y=1.01)

def sfs_label(length, data):
    """
    Set up sfs caption, label and title.

    Parameter
    ---------
    length: length of the SFS
    title: title of the plot
    """
    # Caption
    plt.legend(loc="upper right", fontsize="x-large")

    plt.xlabel("genome wide Frequency", fontsize="xx-large")
    plt.ylabel("number of sites", fontsize="xx-large")
    # Title + show
    title = (
        "SFS  pour différents scénarios avec Ne={}, mu={}, taux de recombinaison={}"
        " & L={:.1e}"
    ).format(data[2]['Ne'], data[2]['mu'], data[2]['ro'], data[2]['length'])
    plt.title(title, fontsize="xx-large", fontweight='bold', y=1.01)


def boxplot_length_mrf(data, title_backup, save=False):
    fig = plt.figure(figsize=(12, 9))
    plt.boxplot(data.values())
    plt.xticks([1, 2, 3], data.keys())
    if save:
        plt.savefig('{}boxplot.png'.format(title_backup), format='png', dpi=150)
    plt.clf()


def plot_dist(type, data, out, save=False):
    """
    Graphic representation of Site Frequency Spectrum (SFS), save to the folder ./Figures.

    Parameter
    ---------
    data: tupple
        - 0: dictionary of {model: ld}
        - 1: dictionary of {model: model_parameter}, with  model_parameter either (tau, kappa)
          or (m12, kappa)
        - 2: dictionary of msprime simulation parameters

    save: boolean
        If set to True save the plot in ./Figures/ld-shape
    """
    color = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:gray"]
    labels = ['Modèle déclin', 'Modèle constant', 'Modèle croissance']

    # Set up plot
    plt.figure(figsize=(12, 9))  #, constrained_layout=True)
    cpt = 0
    for key, elem in data[0].items():
        # Label
        label = key
        for param, value in data[1][key].items():
            label += "{}={}".format(
                'τ' if param == 'Tau' else 'κ' if param == 'Kappa' else param, value
            )
            if param != list(data[1][key].keys())[-1]:
                label += ", "

        with plt.style.context('seaborn-whitegrid'):  # use seaborn style for plot
            plt.plot(elem, color=color[cpt], label=label,
                     marker='o' if key == 'Theoretical model' else '')
        cpt += 1

    # Label, caption and title

    sfs_label(len(elem), data) if type == "sfs" else ld_label(len(elem), data)

    if save:
        plt.savefig('{}.png'.format(title_backup), format='png', dpi=150)
    plt.clf()


def heatmap_axis(ax, xaxis, yaxis, cbar, lrt=False):
    """
    Heatmap customization.

    Parameter
    ---------
    ax: matplotlib.axes.Axes
        ax to modify
    xaxis: str
        x-axis label
    yaxis: str
        y-axis label
    cbar: str
        colormap label
    """
    # Name
    names = [
        "$Log_{10}$" + "({})".format('τ' if xaxis == 'Tau' else xaxis),
        "$Log_{10}$" + "({})".format('κ' if yaxis == 'Kappa' else yaxis)
    ]  # (xaxis, yaxis)

    # x-axis
    plt.xticks(
        np.arange(64, step=7) + 0.5,
        labels=[round(ele, 2) for ele in np.arange(-4, 2.5, 0.7)],
        rotation='horizontal'
    )

   # plt.xticks(
    #    np.arange(3) + 0.5,
     #   labels=[round(ele, 2) for ele in np.arange(-1.0, 2.0, 1.0)],
      #  rotation='horizontal'
    #)
    plt.xlabel(names[0], fontsize="large")

    #y-axis
   # ax.set_ylim(ax.get_ylim()[::-1])  # reverse y-axis
    plt.yticks(
      np.arange(64, step=7) + 0.5,
      labels=[round(ele, 2) for ele in np.arange(-3.5, 3, 0.7)]
    )
    #plt.yticks(
     #   np.arange(3) + 0.5,
      #  labels=[round(ele, 2) for ele in np.arange(-1.0, 2.0, 1.0)]
    #)
    plt.ylabel(names[1], fontsize="large")

    # Set colorbar label & font size
    ax.figure.axes[-1].set_ylabel(cbar, fontsize="large")
    if lrt:
        ax.collections[0].colorbar.set_ticks([0, 20., 40., 60., 80., 100.])
        ax.collections[0].colorbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # Set axis label of kappa = 0.0, i.e. constant model
    for i, ele in enumerate(plt.gca().get_yticklabels()):
        index = [
            i for i, ele in enumerate(plt.gca().get_yticklabels()) if ele.get_text() == "0.0"
        ][0]  # get the index of kappa = 0.0
        plt.gca().get_yticklabels()[index].set_color('#8b1538')  # set the color
        plt.gca().get_yticklabels()[index].set_fontsize('medium')  # set the size
        plt.gca().get_yticklabels()[index].set_fontweight('bold')  # set the weight

    # Hlines for kappa = 0
    ax.hlines([35, 36], *ax.get_xlim(), colors="white", lw=2.)
    ax.vlines([0, 65], ymin=35, ymax=36, color="white", lw=2.)


def plot_heatmap(data, title, cbar, filout="heatmap.png", lrt=False):
    """
    Heatmap

    Parameter
    ---------
    data: pandas DataFrame
    """
    # Set up plot
    plt.figure(figsize=(12, 9), constrained_layout=True)
    sns.set_theme(style='whitegrid')

    # Pre-processing data
    df = data.pivot(index=data.columns[1], columns=data.columns[0], values=data.columns[2])
    # Plot
    ax = sns.heatmap(df, cmap='viridis')

    # Heatmap x and y-axis personnalization
    heatmap_axis(ax=ax, xaxis=df.columns.name, yaxis=df.index.name, cbar=cbar, lrt=lrt)

    # Title
    plt.title(title, fontsize="large", fontweight='bold', pad=10.5)

    plt.savefig(filout, format='png', dpi=150)
    plt.plot()
