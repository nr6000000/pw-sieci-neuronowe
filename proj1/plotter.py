import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import animation
from matplotlib import pyplot as plt


class Plotter:
    @staticmethod
    def plot_losses(loss_history: list[float]) -> None:
        plt.plot(np.arange(len(loss_history)), loss_history)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.title("Training loss")

    @staticmethod
    def plot_weights(
        weight_history: list[list[np.ndarray]],
        bias_history: list[list[np.ndarray]],
        num_frames,
        interval,
    ) -> None:
        def plot_nn(weights, biases, ax):
            layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]

            G = nx.DiGraph()
            pos = {}

            node_colors = []
            for i, layer_size in enumerate(layer_sizes):
                y_offset = (layer_size - 1) / 2.0  # Center the layer vertically
                for j in range(layer_size):
                    node_id = f"L{i}_N{j}"
                    G.add_node(node_id, layer=i)

                    if i == 0:
                        node_colors.append(0)
                    else:
                        node_colors.append(biases[i-1][j])

                    # Assign coordinates: X-axis represents layer index, Y-axis represents neuron position
                    pos[node_id] = (i, j - y_offset)

            edge_colors = []
            for i in range(len(layer_sizes) - 1):
                for j in range(layer_sizes[i]):
                    for k in range(layer_sizes[i + 1]):
                        source = f"L{i}_N{j}"
                        target = f"L{i+1}_N{k}"
                        G.add_edge(source, target)
                        edge_colors.append(weights[i][j][k])

            # Draw nodes
            node_artist = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=node_colors, # type: ignore
                node_size=50,
                cmap=plt.cm.coolwarm,
                vmin=np.quantile(np.concatenate([b.flatten() for b in biases]), 0.1),
                vmax=np.quantile(np.concatenate([b.flatten() for b in biases]), 0.9),
            )

            # Draw edges
            edge_artist = nx.draw_networkx_edges(
                G, 
                pos,
                ax=ax,
                edge_color=edge_colors,  # type: ignore
                width=2, 
                arrows=False, 
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=np.quantile(np.concatenate([w.flatten() for w in weights]), 0.1),
                edge_vmax=np.quantile(np.concatenate([w.flatten() for w in weights]), 0.9),
            )

            return node_artist, edge_artist

        step = len(weight_history)//num_frames
        weight_samples = weight_history[::step]
        bias_samples = bias_history[::step]

        fig, ax = plt.subplots()
        titles = [ax.text(0.5, 1.05, f"Historia wag - batch {i}",
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", 
                    transform=ax.transAxes)
                    for i in range(0, len(weight_history), step)]
        nn_graphs = [plot_nn(weights, biases, ax)
                   for weights, biases in zip(weight_samples, bias_samples)]
        artists = [(title, *nn) for title, nn in zip(titles, nn_graphs)]
        _anim = animation.ArtistAnimation(
            fig=fig, 
            artists=artists, 
            blit=False, 
            interval=interval,
        )

    @staticmethod
    def plot_lineweight(
        weight_history: list[list[np.ndarray]],
        weight_index: tuple[int, int, int],
    ) -> None:
        i, j, k = weight_index
        plt.plot([weights[i][j][k] for weights in weight_history], label=f'w_{i}_{j}_{k}')
        plt.xlabel('Batch')
        plt.ylabel("Value")
        plt.title("Parameter values during training")

    @staticmethod
    def plot_linebias(
        bias_history: list[list[np.ndarray]],
        bias_index: tuple[int, int],
    ):
        i, j = bias_index
        sns.lineplot([biases[i][j] for biases in bias_history], label=f'b_{i}_{j}')
        plt.xlabel('Batch')
        plt.ylabel("Value")
        plt.title("Parameter values during training")

    @staticmethod
    def plot_classification_results(X: np.ndarray, y: np.ndarray, name: str) -> None:
        points = plt.scatter(X[:,0], X[:,1], c=np.asarray(y).ravel())
        plt.legend(*points.legend_elements(num=np.unique(y).tolist()), title="Class")
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title(f"Classification: {name}")

    @staticmethod
    def plot_regression_results(
        X: np.ndarray, y: np.ndarray, dataset: str, label: str, *, points: bool = False
    ) -> None:
        if points:
            plt.scatter(X[:,0], y[:,0], label=label, s=12)
        else:
            order = np.argsort(X[:,0])
            plt.plot(X[order,0], y[order,0], label=label)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title(f"Regression: {dataset}")

