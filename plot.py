import matplotlib.pyplot as plt
import numpy as np

default_colors = [
    "red",
    "green",
    "blue",
    "magenta",
    "pink",
    "orange",
    "beige",
    "violet",
    "gray",
    "yellow",
    "black",
    "purple",
]


def plot_scaling(
    df,
    name_col="name",
    node_col="nodes",
    time_col="time",
    get_color=None,
    get_label=None,
    plot_raw=False,
    plot_scaling=True,
    plot_efficiency=False,
    plot_ideal=True,
    efficiency_params=None,
    ideal_plot_color="orange",
    **kwargs
):
    """
    Plot scaling and efficiency plots based on dataframe containing results of several experiments, 
    where is experiment is run with different nodes.
    Each experiment is referenced by a name, obtained from the column `name_col`, by default `name`.
    Each row of `df` is a result of an experiment (`name_col`) and a given number of nodes (`node_col`), the result
    of the benchmark should be in the column `imagespersec_col`. the column `imagespersec` should contain
    the total images per sec, meaning it should be the number of images per sec per GPU multiplied by the total
    number of GPUs.

    df: pandas Data Frame

    name_col: str
        name of the column that references each experiment
        
    node_col: str
        name of the column that references the number of nodes

    imagespersec_col: str
        name of the column that references the total number of images per sec
    
    get_color: None or function
        if None, use `default_colors`, otherwise,
        it's a function that takes an experiment name as input
        and returns the color of the plot

    get_label: None or function
        if None, use the original experiment name, otherwise
        it's a function that takes an experiment name as input
        and returns the label which will be used in the legend.
        Can be useful to put a set of experiments into the same
        label (e.g., if we want to compare a set of experiments with another
        set of experiments)
    
    plot_raw: bool
        if True, plot the raw images per sec numbers.
        plot_raw and plot_scaling cannot be both true

    plot_scaling: bool
        if True, plot the scaling (also called "speedup").
        plot_raw and plot_scaling cannot be both true

    plot_efficiency: bool
        whether to plot the efficiency plot (reference: https://arxiv.org/pdf/1807.09161.pdf)
    
    plot_ideal: bool
        if True, plot the ideal curve, it will be different wheether `plot_scaling` is True or False.
        if `plot_scaling` is `True`, it's just the identity function f(node)=node w.r.t to number of nodes.
        if `plot_scaling` is `False`, for each experiment `name`, we plot f(node)=baseline*node, where baseline
        is images per sec for the smallest number of nodes.

    efficiency_params: dict
        params for plotting efficiency.
        list of params (keys of the dict):
        - text_size (default: 11), size of text annotation
        - text_spacing (default: 0.1), space between plot and text  annotation
        - scale (default: 20), height of the efficiency plot
        - top (default:0), starting y coordinate of the efficiency plot 
        - ymin and ymax: alternatively to specify scale, specify min y position (ymin)
          and max y position for efficiency plot
    **kwargs: kwargs given to plt.figure()

    Returns
    -------
    
    df (pandas DataFrame)

    a copy of `df`, where the columns `scaling`, `baseline`, and `efficiency` are
    added.
    """
    assert not (plot_raw and plot_scaling), "plot_raw and plot_scaling cannot be both true."

    fig = plt.figure(**kwargs)
    df = df.copy()
    df = df.set_index(name_col)
    df["baseline"] = None

    min_node = df[node_col].min()
    # Compute "scaling" and "efficiency" for each experiment (determined by the col `name_col`)
    for name, group in df.groupby(df.index):
        group = group.sort_values(by=node_col)
        baseline = group[time_col].values[0]
        df.loc[name, "baseline"] = baseline
    k = (df[node_col]/min_node)
    df["scaling"] = (df["baseline"] / df[time_col]) * k
    df["efficiency"] =  df["baseline"] / df[time_col]
    if efficiency_params is None:
        efficiency_params = {}
    # Plot scaling (and optionally efficiency) of each experiment (determined the col `name_col`)
    for i, (name, group) in enumerate(df.groupby(df.index)):
        group = group.sort_values(by=node_col)
        if get_color and get_color(name):
            c = get_color(name)
        else:
            c = default_colors[i]
        if get_label and get_label(name):
            l = get_label(name)
            #to avoid adding multiple times the same label
            l = l if l not in plt.gca().get_legend_handles_labels()[1] else ''
        else:
            l = name

        nodes = group[node_col].values
        if plot_scaling:
            scaling = group["scaling"].values
            plt.plot(nodes, group.scaling, c=c, label=l, marker="o")
        elif plot_raw:
            imagespersec = group[time_col].values * df[node_col]
            plt.plot(nodes, imagespersec, c=c, label=l, marker="o")

        if plot_efficiency:
            text_size = efficiency_params.get("text_size", 11)
            scale = efficiency_params.get("scale", 20)
            top = efficiency_params.get("top", 0)
            text_spacing =  efficiency_params.get("text_spacing", 0.1)
            eff = group["efficiency"].values 
            ymin = efficiency_params.get("ymin")
            ymax = efficiency_params.get("ymax")
            normalize = efficiency_params.get("normalize", True)
            y = eff
            if normalize:
                y = (y - y.min()) / (y.max() - y.min())
            if ymin is not None and ymax is not None:
                y = y * (ymax - ymin) + ymin
            else:
                y = eff*scale+top
            plt.plot(group[node_col], y, c=c, marker="o", label=l if not plot_scaling and not plot_raw else None)
            # add text annotation for efficiency plot
            for node, effval, yval in zip(nodes, eff, y):
                plt.text(node, yval+text_spacing, f"{effval*100:.2f}%", size=text_size, c=c)
    plt.xlabel("nodes")

    if plot_scaling:
        plt.ylabel("scaling")
    else:
        plt.ylabel("images per sec")
    
    if plot_ideal:
        # ideal
        if plot_scaling:
            #if scaling plot, it's just the identity function
            nodes = df[node_col].unique()
            min_node = nodes.min()
            ideal_scaling = nodes / min_node 
            plt.plot(nodes, ideal_scaling, c=ideal_plot_color, label="ideal", marker="o")
        else:
            #if raw imagespersec, figure plot an ideal plot for each experiment `name`
            for i, (name, group) in enumerate(df.groupby(df.index)):
                nodes = group[node_col].unique()
                baseline = group.baseline.values[0]
                plt.plot(nodes, baseline*nodes, c=ideal_plot_color, label=f"{name}_ideal", marker="o")
    plt.xticks(nodes)
    plt.legend()
    return df.reset_index()

