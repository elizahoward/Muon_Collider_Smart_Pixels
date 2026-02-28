import hls4ml
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np
import numpy.typing as npt
import re
import pandas as pd
import optuna

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from typing import List, Callable
from paretoset import paretoset


class Draw:
    def __init__(self, output_dir: Path = Path("plots"), interactive: bool = False):
        self.output_dir = output_dir
        self.interactive = interactive
        self.cmap = ["green", "red", "blue", "orange", "purple", "brown"]
        hep.style.use("CMS")

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def _save_fig(self, name: str, bbox_extra_artists=None, bbox_inches='tight') -> None:
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.png",
            bbox_inches=bbox_inches, 
            bbox_extra_artists=bbox_extra_artists
        )
        if self.interactive:
            plt.show()
        plt.close()

    def plot_loss_history(
        self, training_loss: npt.NDArray, validation_loss: npt.NDArray, name: str
    ):
        plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label="Training")
        plt.plot(
            np.arange(1, len(validation_loss) + 1), validation_loss, label="Validation"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0, 10)
        self._save_fig(name)

    def plot_loss_histories(
        self, loss_dict: dict[str, (npt.NDArray, npt.NDArray)], name: str
    ):
        for model_name, (train_loss, val_loss) in loss_dict.items():
            c = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(np.arange(1, len(train_loss) + 1), train_loss, color=c, label=f"{model_name} (Training)")
            plt.plot(np.arange(1, len(val_loss) + 1), val_loss, color=c, ls=":", label=f"{model_name} (Validation)")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_regional_deposits(self, deposits: npt.NDArray, mean: float, name: str):
        im = plt.imshow(
            deposits.reshape(18, 14), vmin=0, vmax=deposits.max(), cmap="Purples"
        )
        ax = plt.gca()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r"Calorimeter E$_T$ deposit (GeV)")
        plt.xticks(np.arange(14), labels=np.arange(4, 18))
        plt.yticks(
            np.arange(18),
            labels=np.arange(18)[::-1],
            rotation=90,
            va="center",
        )
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")
        plt.title(rf"Mean E$_T$ {mean: .2f} ({name})")
        self._save_fig(f'profiling-mean-deposits-{name}')

    def plot_spacial_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str, apply_weights: bool = False
    ):
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        for deposit, label in zip(deposits, labels):
            bins = np.argwhere(deposit)
            phi, eta = bins[:, 1], bins[:, 2]
            if apply_weights:
                weights = deposit[np.nonzero(deposit)]
            else:
                weights = np.ones(phi.shape)
            ax1.hist(
                eta + 4,
                weights=weights,
                density=True,
                facecolor=None,
                bins=np.arange(4, 19),
                label=label,
                histtype="step"
            )
            ax2.hist(
                phi,
                weights=weights,
                density=True,
                facecolor=None,
                bins=np.arange(19),
                label=label,
                histtype="step",
            )
        ax1.set_ylabel("a.u.")
        ax1.set_xlabel(r"i$\eta$")
        ax2.set_xlabel(r"i$\phi$")
        plt.legend(loc="best")
        self._save_fig(f'profiling-spacial-{name}')

    def plot_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str
    ):
        for deposit, label in zip(deposits, labels):
            plt.hist(
                deposit.reshape((-1)),
                bins=100,
                range=(0, 1024),
                density=1,
                label=label,
                log=True,
                histtype="step",
            )
        plt.xlabel(r"E$_T$")
        plt.legend(loc="best")
        self._save_fig(f'profiling-deposits-{name}')

    def plot_reconstruction_results(
        self,
        deposits_in: npt.NDArray,
        deposits_out: npt.NDArray,
        loss: float,
        name: str,
    ):
        fig, (ax1, ax2, ax3, cax) = plt.subplots(
            ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = max(deposits_in.max(), deposits_out.max())

        ax1 = plt.subplot(1, 4, 1)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title("Original", fontsize=18)
        ax1.imshow(
            deposits_in.reshape(18, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax2 = plt.subplot(1, 4, 2)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title("Reconstructed", fontsize=18)
        ax2.imshow(
            deposits_out.reshape(18, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax3 = plt.subplot(1, 4, 3)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(rf"|$\Delta$|, MSE: {loss: .2f}", fontsize=18)

        im = ax3.imshow(
            np.abs(deposits_in - deposits_out).reshape(18, 14),
            vmin=0,
            vmax=max_deposit,
            cmap="Purples",
        )

        ip = InsetPosition(ax3, [1.05, 0, 0.05, 1])
        cax.set_axes_locator(ip)
        fig.colorbar(im, cax=cax, ax=[ax1, ax2, ax3]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )
        self._save_fig(name)

    def plot_phi_shift_variance(
        self, losses: List[float], name: str
    ):
        x = np.arange(len(losses))
        loss_means = np.mean(losses, axis=1)
        plt.plot(x, loss_means)
        loss_stds =  np.std(losses, axis=1)
        lower = loss_means - loss_stds / 2
        upper = loss_means + loss_stds / 2
        plt.fill_between(x, lower, upper, alpha=0.1)
        plt.xlabel(r"Shift [$\Delta$ i$\phi$]")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(r"$\Delta_{rel} (MSE)$")
        plt.axvline(x=0, color='grey', linestyle=':', label='Original')
        plt.axvline(x=18, color='grey', linestyle=':')
        plt.axhline(y=loss_means[0], color='grey', linestyle=':')
        plt.legend()
        self._save_fig(name)

    def plot_anomaly_score_distribution(
        self, scores: List[npt.NDArray], labels: List[str], name: str
    ):
        for score, label in zip(scores, labels):
            plt.hist(
                score.reshape((-1)),
                bins=100,
                range=(0, 256),
                density=1,
                label=label,
                log=True,
                histtype="step",
            )
        plt.xlabel(r"Anomaly Score")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self._save_fig(name)

    def plot_roc_curve(
        self,
        y_trues: List[npt.NDArray],
        y_preds: List[npt.NDArray],
        labels: List[str],
        inputs: List[npt.NDArray],
        name: str,
        cv: int = 3,
    ):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for y_true, y_pred, label, color, d in zip(
            y_trues, y_preds, labels, self.cmap, inputs
        ):
            aucs = []
            for _, indices in skf.split(y_pred, y_true):
                fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
                aucs.append(auc(fpr, tpr))
            std_auc = np.std(aucs)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            fpr_base, tpr_base, _ = roc_curve(y_true, np.mean(d**2, axis=(1, 2)))

            plt.plot(
                fpr * 28.61,
                tpr,
                linestyle="-",
                lw=1.5,
                color=color,
                alpha=0.8,
                label=rf"{label} (AUC = {roc_auc: .4f} $\pm$ {std_auc: .4f})",
            )

            plt.plot(
                fpr_base * 28.61,
                tpr_base,
                linestyle="--",
                lw=1.0,
                color=color,
                alpha=0.5,
                label=rf"{label}, Baseline",
            )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            lw=1,
            color="black",
            label="3 kHz",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self._save_fig(name)
    
    def get_aucs(
        self,
        y_trues: List[npt.NDArray],
        y_preds: List[npt.NDArray],
        max_rate: float = 0.003/28.61,
        min_rate: float = 0.0003/28.61,
        use_cut_rate: bool = False,
        cv: int = 3,
    ):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        std_aucs, roc_aucs = [], []
        for y_true, y_pred in zip(
            y_trues, y_preds
        ):
            aucs = []
            for _, indices in skf.split(y_pred, y_true):
                fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
                aucs.append(auc(fpr, tpr))
            std_aucs.append(np.std(aucs))

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            fpr = fpr.flatten()
            tpr = tpr.flatten()
            if use_cut_rate:
                min_ind = np.searchsorted(fpr, min_rate, side='right')
                max_ind = np.searchsorted(fpr, max_rate, side='right')
                tpr = tpr[min_ind:max_ind]
                fpr = fpr[min_ind:max_ind+1]
                tpr = np.clip(tpr, 0.01, 1.)
                tpr = np.log10(tpr)
                tpr = tpr+2
                tpr = tpr/2.
                fpr = np.log10(fpr)
                fpr = np.diff(fpr)
                width = np.sum(fpr)
                if width==0: 
                    to_append = 0.
                else:
                    to_append = np.sum(np.dot(tpr, fpr))/width
                roc_aucs.append(to_append)
            else: roc_aucs.append(auc(fpr, tpr))
        return roc_aucs, std_aucs

    def plot_compilation_error(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.scatter(scores_keras, np.abs(scores_keras - scores_hls4ml), s=1)
        plt.xlabel("Anomaly Score, $S$")
        plt.ylabel("Error, $|S_{Keras} - S_{hls4ml}|$")
        self._save_fig(f'compilation-error-{name}')

    def plot_compilation_error_distribution(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.hist(scores_keras - scores_hls4ml, fc="none", histtype="step", bins=100)
        plt.xlabel("Error, $S_{Keras} - S_{hls4ml}$")
        plt.ylabel("Number of samples")
        plt.yscale("log")
        self._save_fig(f'compilation-error-dist-{name}')

    def plot_cpp_model(self, hls_model, name: str):
        hls4ml.utils.plot_model(
            hls_model,
            show_shapes=True,
            show_precision=True,
            to_file=f"{self.output_dir}/cpp-model-{self._parse_name(name)}.png",
        )

    def plot_roc_curve_comparison(
        self, scores_keras: dict, scores_hls4ml: npt.NDArray, name: str
    ):
        fpr_model: list = []
        tpr_model: list = []

        scores_keras_normal = scores_keras["Background"]
        scores_hls4ml_normal = scores_hls4ml["Background"]

        for dataset_name, color in zip(list(scores_keras.keys())[:-1], self.cmap):
            scores_keras_anomaly = scores_keras[dataset_name]
            scores_hls4ml_anomaly = scores_hls4ml[dataset_name]

            y_true = np.append(
                np.zeros(len(scores_keras_normal)), np.ones(len(scores_hls4ml_anomaly))
            )
            y_score_keras = np.append(scores_keras_normal, scores_keras_anomaly)
            y_score_hls = np.append(scores_hls4ml_normal, scores_hls4ml_anomaly)

            for y_scores, model, ls in zip(
                [y_score_keras, y_score_hls], ["Keras", "hls4ml"], ["-", "--"]
            ):
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                plt.plot(
                    fpr * 28.61,
                    tpr,
                    linestyle=ls,
                    color=color,
                    label="{0}: {1}, AUC = {2:.4f}".format(
                        model, dataset_name, auc(fpr, tpr)
                    ),
                )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            color="black",
            label="3 kHz trigger rate",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self._save_fig(f'compilation-roc-{name}')

    def plot_output_reference(self):
        with open("misc/output-reference.txt") as f:
            data = f.read()
        data = np.array([row.split(",") for row in data.split("\n")[:-1]]).astype(
            np.int8
        )
        data = np.flipud(data) - 1
        legend_elements = [
            Patch(
                facecolor=self.cmap[0],
                edgecolor=self.cmap[0],
                label="Anomaly Detection, Integer Part",
            ),
            Patch(
                facecolor=self.cmap[1],
                edgecolor=self.cmap[1],
                label="Anomaly Detection, Decimal Part",
            ),
            Patch(
                facecolor=self.cmap[2], edgecolor=self.cmap[2], label="Heavy Ion Bit"
            ),
            Patch(facecolor=self.cmap[3], edgecolor=self.cmap[3], label="Reserved"),
        ]
        plt.figure(figsize=(25, 5))
        plt.pcolor(
            data, edgecolors="black", alpha=0.6, cmap=ListedColormap(self.cmap[:4])
        )
        plt.xticks([])
        plt.yticks([])
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    abs(y * 32 + x - 191),
                    horizontalalignment="center",
                    fontsize=16,
                    verticalalignment="center",
                )
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0, 0),
            loc="upper left",
            frameon=False,
            ncol=4,
            borderaxespad=0,
        )
        self._save_fig('ugt-link-reference')

    def plot_results_supervised(
        self, grid: npt.NDArray, models: list[str], datasets: list[str], name: str
    ):
        plt.imshow(grid, alpha=0.7, cmap="RdYlGn")
        plt.xticks(
            np.arange(len(models)),
            labels=models,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.yticks(np.arange(len(datasets)), labels=datasets)
        for i in range(len(datasets)):
            for j in range(len(models)):
                text = plt.text(
                    j,
                    i,
                    "{0:.3f}".format(grid[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    size=16,
                )
        self._save_fig(f'supervised-{name}')

    def make_equivariance_plot(
        self,
        image: npt.NDArray,
        f: Callable[npt.NDArray, npt.NDArray],  # symmetry transformation
        g: Callable[npt.NDArray, npt.NDArray],  # mapping of the model
        name: str
    ):

        fig, axs = plt.subplots(
            nrows=2, ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = image.max()
        xmax, ymax, _ = image.shape

        mse_g_1 = float(np.mean((g(image) - image)**2))
        mse_gf_f = float(np.mean((g(f(image)) - f(image))**2))
        mse_gf_fg = float(np.mean((g(f(image)) - f(g(image)))**2))

        axs[0, 0].imshow(image, vmin=0, vmax=max_deposit, cmap="Purples")
        axs[0, 1].imshow(f(image), vmin=0, vmax=max_deposit, cmap="Purples")
        im = axs[0, 2].imshow(g(f(image)), vmin=0, vmax=max_deposit, cmap="Purples")
        ip = InsetPosition(axs[0][2], [1.05, 0, 0.05, 1])
        axs[0][3].set_axes_locator(ip)
        fig.colorbar(im, cax=axs[0][3], ax=axs[0][:-1]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        axs[1, 0].imshow(image, vmin=0, vmax=max_deposit, cmap="Purples")
        axs[1, 1].imshow(g(image), vmin=0, vmax=max_deposit, cmap="Purples")
        im = axs[1, 2].imshow(f(g(image)), vmin=0, vmax=max_deposit, cmap="Purples")
        ip = InsetPosition(axs[1][2], [1.05, 0, 0.05, 1])
        axs[1][3].set_axes_locator(ip)
        fig.colorbar(im, cax=axs[1][3], ax=axs[1][:-1]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        axs[0, 0].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 0].text(xmax-3.5, ymax/2+1, 'trans', fontsize=18)

        axs[0, 1].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 1].text(xmax-3.5, ymax/2+1, 'pred', fontsize=18)
        axs[0, 1].text(xmax-4, ymax/2+3, rf"MSE: {mse_gf_f:.1f}", fontsize=16)

        axs[1, 0].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 0].text(xmax-3.5, ymax/2+1, 'pred', fontsize=18)
        axs[1, 0].text(xmax-4, ymax/2+3, rf"MSE: {mse_g_1:.1f}", fontsize=16)

        axs[1, 1].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 1].text(xmax-3.5, ymax/2+1, 'trans', fontsize=18)

        axs[0, 2].annotate('', xy=(0.5, -0.2), xycoords='axes fraction', 
                           xytext=(0.5, 0), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='<->'))
        axs[0, 2].text(xmax/2-1.5, ymax+6, rf"MSE: {mse_gf_fg:.2f}", fontsize=16)

        for row in axs:
            for ax in row[:-1]:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        self._save_fig(name)

    def plot_2d(
            self, 
            x: npt.NDArray, 
            y: npt.NDArray, 
            xerr: npt.NDArray = None, 
            yerr: npt.NDArray = None, 
            xlabel: str = 'Objective 0', 
            ylabel: str = 'Objective 1', 
            to_enumerate: list = [], 
            label_seeds: bool = True, 
            name: str = 'example_objectives', 
    ):
        x=np.reshape(np.array(x), (-1))
        y=np.reshape(np.array(y), (-1))

        plt.scatter(x, y, label=f'n = {x.shape[0]}', color='black')
        for i in range(len(to_enumerate)):
            plt.annotate(to_enumerate[i], (x[i], y[i]), size = 15, xytext=(-2.5, 0.5), textcoords='offset fontsize')
        if label_seeds:
            for i in range(len(x)):
                plt.annotate(i, (x[i], y[i]), size = 10, xytext = (0, -1.5), textcoords = 'offset fontsize')
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f'{xlabel} vs {ylabel}')

        self._save_fig(name)
        plt.clf()

    def plot_2d_pareto(
        self, 
        argname, 
        name_x: str, 
        name_y: str, 
        num_trials_in_study: list = [],
        trial_names: list = [], 
        min_pareto_length: int = 0, 
        to_enumerate: list = [], 
        label_seeds: bool = True, 
        show_non_pareto: bool = False, 
        show_legend: bool = True, 
        zoom: bool = False, 
        name: str = 'example_objectives'
    ):
        if type(argname) == str:
            x = [np.load(f'arch/{argname}/trial_metrics/{name_x}/{trial_names[i]}').flatten() for i in range(len(trial_names))]
            y = [np.load(f'arch/{argname}/trial_metrics/{name_y}/{trial_names[i]}').flatten() for i in range(len(trial_names))]
        elif type(argname) == list:
            x, y = [], []
            num_trials_in_study_diff = np.cumsum(num_trials_in_study)
            num_trials_in_study_diff = np.concatenate((np.array([0]), num_trials_in_study_diff))
            for i in range(len(argname)):
                x = x + [np.load(f'arch/{argname[i]}/trial_metrics/{name_x}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
                y = y + [np.load(f'arch/{argname[i]}/trial_metrics/{name_y}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]

        # Optimization direction
        if "AUC" in name_x and "Loss" in name_y:
            op_x, op_y = "max", "min"
        elif "AUC" in name_y and "Loss" in name_x:
            op_x, op_y = "min", "max"

        # Compute Pareto sets
        data = [pd.DataFrame({name_x: x[i], name_y: y[i]}) for i in range(len(trial_names))]
        mask = [paretoset(data[i], sense=[op_x, op_y]) for i in range(len(trial_names))]
        for i in range(len(mask)):
            if mask[i].sum() < min_pareto_length:
                mask[i] = pd.Series([False] * len(mask[i]), index=data[i].index)
        pareto_data = [data[i][mask[i]] for i in range(len(trial_names))]

        x_pareto = [pareto_data[i].get(name_x).to_numpy().flatten() for i in range(len(trial_names))]
        y_pareto = [pareto_data[i].get(name_y).to_numpy().flatten() for i in range(len(trial_names))]
        trial_names = [name.replace(".npy", "") for name in trial_names]

        # Define colormap (x values → viridis colormap)
        cmap = mpl.colormaps['viridis'].resampled(len(trial_names))

        n = 0
        # Plot Pareto and non-Pareto points
        for i in range(len(trial_names)):
            color = cmap(i / len(trial_names))  # Map trial index to color

            # Sort Pareto points by x for better visualization
            sorted_indices = np.argsort(x_pareto[i])
            x_p_sorted, y_p_sorted = x_pareto[i][sorted_indices], y_pareto[i][sorted_indices]

            # Plot Pareto front (larger markers, black edges)
            plt.scatter(
                x_p_sorted, y_p_sorted, 
                s=20, c='gray', edgecolors='black', label=f'Pareto {trial_names[i]}', alpha=0.3
            )
            n += len(x_p_sorted)

            # Connect Pareto points with lines
            #plt.plot(
            #    x_p_sorted, y_p_sorted, 
            #    color=color, linestyle='-', linewidth=2, alpha=0.7
            #)

            if show_non_pareto:
                # Non-Pareto points (smaller markers, gray edges)
                plt.scatter(
                    x[i], y[i], 
                    s=40, c=[color], edgecolors='gray', alpha=0.5, label=f'All {trial_names[i]}'
                )
                n += len(x[i])

                if label_seeds:
                    for j in range(len(x[i])):
                        plt.annotate(j, (x[i][j], y[i][j]), size=10, xytext=(0, -1.5), textcoords='offset fontsize')

        # Annotate special points
        for i in range(len(to_enumerate)):
            plt.annotate(to_enumerate[i], (x[0][i], y[0][i]), size=15, xytext=(-2.5, 0.5), textcoords='offset fontsize')

        plt.xlabel(name_x)
        plt.ylabel(name_y)
        if show_legend:
            plt.legend()
        if zoom: plt.xlim(0, 10)
        else: plt.xlim(0, 40)
        plt.ylim(-0.1, 0.9)        
        plt.title(f'{name_x} vs {name_y}, n={n}')
        
        self._save_fig(name)
        plt.clf()

    def plot_3d_pareto_executions(
        self, 
        argname, 
        name_x: str, 
        name_y: str, 
        name_z: str,
        num_trials_in_study: list = [],
        trial_names: list = [], 
        min_pareto_length: int = 0, 
        to_enumerate: list = [], 
        label_seeds: bool = True, 
        zoom: bool = False,
        name: str = 'example_objectives_3d'
    ):
        if type(argname) == str:
            x = [np.load(f'arch/{argname}/trial_metrics/{name_x}/{trial_names[i]}').flatten() for i in range(len(trial_names))]
            y = [np.load(f'arch/{argname}/trial_metrics/{name_y}/{trial_names[i]}').flatten() for i in range(len(trial_names))]
            z = [np.load(f'arch/{argname}/trial_metrics/{name_z}/{trial_names[i]}').flatten() for i in range(len(trial_names))]
        elif type(argname) == list:
            x, y, z = [], [], []
            num_trials_in_study_diff = np.cumsum(num_trials_in_study)
            num_trials_in_study_diff = np.concatenate((np.array([0]), num_trials_in_study_diff))
            for i in range(len(argname)):
                x = x + [np.load(f'arch/{argname[i]}/trial_metrics/{name_x}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
                y = y + [np.load(f'arch/{argname[i]}/trial_metrics/{name_y}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
                z = z + [np.load(f'arch/{argname[i]}/trial_metrics/{name_z}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
        
        # Optimization direction
        if "AUC" in name_x and "Loss" in name_y:
            op_x, op_y = "max", "min"
        elif "AUC" in name_y and "Loss" in name_x:
            op_x, op_y = "min", "max"
        op_z = "min"

        # Compute Pareto sets
        all_data = []
        for i in range(len(trial_names)):
            df = pd.DataFrame({
                name_x: x[i],
                name_y: y[i],
                name_z: z[i],
                'trial': trial_names[i],
                'index_in_trial': list(range(len(x[i])))
            })
            all_data.append(df)
        combined_data = pd.concat(all_data, ignore_index=True)

        mask = paretoset(combined_data[[name_x, name_y, name_z]], sense=[op_x, op_y, op_z])
        if mask.sum() < min_pareto_length:
            mask[:] = False
        pareto_data = combined_data[mask]

        grouped = pareto_data.groupby('trial')

        x_pareto = []
        y_pareto = []
        z_pareto = []

        for trial_name in trial_names:
            df = grouped.get_group(trial_name) if trial_name in grouped.groups else pd.DataFrame({name_x: [], name_y: [], name_z: []})
            x_pareto.append(df[name_x].to_numpy())
            y_pareto.append(df[name_y].to_numpy())
            z_pareto.append(df[name_z].to_numpy())

        # Plotting
        fig=plt.figure()
        ax=fig.add_subplot()

        # Combine all Pareto points for colormap normalization
        all_z_vals = np.concatenate([z.flatten() for z in z_pareto])
        norm = mpl.colors.Normalize(vmin=np.min(all_z_vals), vmax=np.max(all_z_vals))
        cmap = mpl.cm.viridis
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        n = 0
        pareto_trial_names = []
        for i in range(len(trial_names)):
            if len(z_pareto[i]) == 0: continue
            sorted_indices = np.argsort(x_pareto[i])
            x_p_sorted, y_p_sorted, z_p_sorted = x_pareto[i][sorted_indices], y_pareto[i][sorted_indices], z_pareto[i][sorted_indices]

            colors = cmap(norm(z_p_sorted))
            ax.scatter(
                x_p_sorted, y_p_sorted,
                c=colors, s=100, edgecolors='black', alpha=0.8
            )
            n += len(x_pareto[i])
            pareto_trial_names.append(int(trial_names[i].replace('.npy', '')))

            # Connect Pareto points with lines
            plt.plot(
                x_p_sorted, y_p_sorted, 
                color=colors[0], linestyle='-', linewidth=2, alpha=0.7
            )
            
        # Annotate selected points
        for i in range(len(to_enumerate)):
            ax.text(x[0][i], y[0][i], z[0][i], str(to_enumerate[i]), size=10)

        ax.set_xlabel(name_x)
        ax.set_ylabel(name_y)
        if zoom: ax.set_xlim(0, 10)
        else: ax.set_xlim(0, 40)
        ax.set_ylim(-0.1, 0.9)
        ax.set_title(f'{name_x} vs {name_y} vs {name_z}, n={n}')
        fig.colorbar(mappable, ax=ax, label=name_z)

        self._save_fig(name)
        plt.clf()

    def get_pareto_executions(
        self, 
        argname,
        study,
        name_x: str,
        name_y: str, 
        name_z: str, 
        trial_names: list = [],
        num_trials_in_study: list = [],
        min_pareto_length: int = 0
        ):
        """
        Return Pareto-optimal executions based on roc_auc (maximize), loss (minimize), and size (minimize).
        Each result includes the metrics and associated hyperparameters.
        """

        # Define objective directions
        if "AUC" in name_x and "Loss" in name_y:
            op_x, op_y = "max", "min"
        elif "AUC" in name_y and "Loss" in name_x:
            op_x, op_y = "min", "max"
        op_z = "min"
        
        if type(argname) == str:
            x = [np.load(f'arch/{argname}/trial_metrics/{name_x}/{t}').flatten() for t in trial_names]
            y = [np.load(f'arch/{argname}/trial_metrics/{name_y}/{t}').flatten() for t in trial_names]
            z = [np.load(f'arch/{argname}/trial_metrics/{name_z}/{t}').flatten() for t in trial_names]
        elif type(argname) == list:
            x, y, z = [], [], []
            num_trials_in_study_diff = np.cumsum(num_trials_in_study)
            num_trials_in_study_diff = np.concatenate((np.array([0]), num_trials_in_study_diff))
            for i in range(len(argname)):
                x = x + [np.load(f'arch/{argname[i]}/trial_metrics/{name_x}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
                y = y + [np.load(f'arch/{argname[i]}/trial_metrics/{name_y}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
                z = z + [np.load(f'arch/{argname[i]}/trial_metrics/{name_z}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten() for j in range(num_trials_in_study[i])]
        
        # Combine data across all trials into a single DataFrame
        all_data = []
        for i, trial in enumerate(trial_names):
            df = pd.DataFrame({
                name_x: x[i],
                name_y: y[i],
                name_z: z[i],
                'trial': trial,
                'index_in_trial': list(range(len(x[i])))
            })
            all_data.append(df)
        combined_data = pd.concat(all_data, ignore_index=True)

        # Apply Pareto front filter
        mask = paretoset(combined_data[[name_x, name_y, name_z]], sense=[op_x, op_y, op_z])
        if mask.sum() < min_pareto_length:
            mask[:] = False
        pareto_data = combined_data[mask]

        # Load trials
        if type(argname) == str:
            trials=study.get_trials()
        if type(argname) == list:
            trials=[]
            for i in range(len(argname)):
                trials=trials+study[i].get_trials()

        output = []
        for _, row in pareto_data.iterrows():
            trial_name = row['trial']
            index = int(row['index_in_trial'])

            # Match trial ID from trial name (assumes name format "trial_{trial_id}")
            trial_id = int(trial_name.replace('.npy', ''))
            trial = trials[trial_id]

            entry = [
                row[name_x],
                row[name_y],
                row[name_z],
                trial.params  # Dictionary of hyperparameters
            ]
            output.append(entry)

        return output

    def plot_3d(
            self, 
            x: npt.NDArray, 
            y: npt.NDArray, 
            z: npt.NDArray, 
            xerr: npt.NDArray = None, 
            yerr: npt.NDArray = None, 
            zerr: npt.NDArray = None, 
            xlabel: str = 'Objective 0', 
            ylabel: str = 'Objective 1', 
            zlabel: str = 'Objective 2', 
            to_enumerate: list = [], 
            label_seeds: bool = True, 
            name: str = 'example_objectives', 
    ):
        x=np.reshape(np.array(x), (-1))
        y=np.reshape(np.array(y), (-1))
        z=np.reshape(np.array(z), (-1))
        zmax=np.max(z)/1500.
        z=z/zmax

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c='black', s=z, alpha=0.5, label=f'n = {x.shape[0]}')
        for i in range(len(to_enumerate)):
            ax.annotate(to_enumerate[i], (x[i], y[i]), size = 15, xytext=(-2.5, 0.5), textcoords='offset fontsize')
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
        for i in range(len(labels)):
            temp = int(float(re.search(r'\d+', labels[i]).group()) * zmax)
            labels[i] = f'$\\mathdefault{{{temp}}}$'
        if label_seeds:
            for i in range(len(x)):
                ax.annotate(i, (x[i], y[i]), size = 10, xytext = (0, -1.5), textcoords = 'offset fontsize')
        ax.legend()
        legend1 = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title=f'{zlabel}', title_fontsize=20, fontsize=15)
        ax.errorbar(x, y, c='black', xerr=xerr, yerr=yerr, fmt='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel} vs {zlabel}, n = {x.shape[0]}')
        
        self._save_fig(name)
        plt.clf()

    def plot_3d_pareto(
            self, 
            name_a: str, 
            name_b: str, 
            name_c: str, 
            std_name_a: str, 
            std_name_b: str, 
            std_name_c: str, 
            argname: str, 
            to_enumerate: list = [], 
            show_non_pareto: bool = False, 
            label_seeds: bool = True, 
            name: str = 'example_objectives', 
    ):
        x = np.load(f'arch/{argname}/study_metrics/{name_a}').flatten()
        y = np.load(f'arch/{argname}/study_metrics/{name_b}').flatten()
        z = np.load(f'arch/{argname}/study_metrics/{name_c}').flatten()

        if "parameters" in name_c:
            z = z / 1000
            name_c = name_c.replace("number", "1000s")
        if "(b)" in name_c:
            z = z / 1024
            name_c = name_c.replace("(b)", "(kb)")

        x_all, y_all = x.copy(), y.copy()

        std_x = np.load(f'arch/{argname}/study_metrics/{std_name_a}').flatten() if std_name_a else np.zeros(len(x))
        std_y = np.load(f'arch/{argname}/study_metrics/{std_name_b}').flatten() if std_name_b else np.zeros(len(y))
        std_z = np.load(f'arch/{argname}/study_metrics/{std_name_c}').flatten() if std_name_c else np.zeros(len(z))

        op_x, op_y, op_z = ("max", "min", "min") if "AUC" in name_a and "Loss" in name_b else ("min", "max", "min")

        data = pd.DataFrame({name_a: x, name_b: y, name_c: z})
        std = pd.DataFrame({std_name_a: std_x, std_name_b: std_y, std_name_c: std_z})

        mask = paretoset(data, sense=[op_x, op_y, op_z])
        pareto_data = data[mask]
        pareto_std = std[mask]

        # Get Pareto front points
        x_pareto, y_pareto, z_pareto = pareto_data[name_a].to_numpy(), pareto_data[name_b].to_numpy(), pareto_data[name_c].to_numpy()
        std_x_pareto, std_y_pareto = pareto_std[std_name_a].to_numpy(), pareto_std[std_name_b].to_numpy()

        # Get non-Pareto front points
        x_non_pareto, y_non_pareto, z_non_pareto = x[~mask], y[~mask], z[~mask]
        std_x_non_pareto, std_y_non_pareto = std_x[~mask], std_y[~mask]

        # Define colormap for z-axis values
        norm = mcolors.Normalize(vmin=min(z), vmax=max(z))
        cmap = cm.viridis

        # Define sizes: Large for Pareto points, small for others
        size_pareto = 150
        size_non_pareto = 50

        fig, ax = plt.subplots()

        # Plot Pareto front points (color by z value, size large)
        ax.scatter(x_pareto, y_pareto, c=cmap(norm(z_pareto)), s=size_pareto, label=f'Pareto: n = {len(x_pareto)}', alpha=0.8, edgecolors='black')
        n = len(x_pareto)

        if show_non_pareto:
            # Plot non-Pareto points (color by z value, size small)
            ax.scatter(x_non_pareto, y_non_pareto, c=cmap(norm(z_non_pareto)), s=size_non_pareto, label=f'All: n = {len(x)}', alpha=0.5, edgecolors='gray')
            n += len(x_non_pareto)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(name_c.replace(".npy", ""), fontsize=12)

        # Annotate points if needed
        if label_seeds:
            for i in range(len(x_all)):
                ax.annotate(i, (x_all[i], y_all[i]), size=10, xytext=(0, 0), textcoords="offset points")

        # Error bars
        ax.errorbar(x_non_pareto, y_non_pareto, xerr=std_x_non_pareto, yerr=std_y_non_pareto, fmt='none', color='black', alpha=0.5)
        ax.errorbar(x_pareto, y_pareto, xerr=std_x_pareto, yerr=std_y_pareto, fmt='none', color='black', alpha=0.8)

        ax.set_xlabel(name_a.replace(".npy", ""))
        ax.set_ylabel(name_b.replace(".npy", ""))
        ax.legend()
        ax.set_xlim(0, 40)
        ax.set_ylim(-0.1, 0.9)
        ax.set_title(f'{name_a.replace(".npy", "")} vs {name_b.replace(".npy", "")} vs {name_c.replace(".npy", "")}, n={n}')

        self._save_fig(name)
        plt.clf()

    def get_3d_pareto(
            self, 
            name_a: str, 
            name_b: str, 
            name_c: str, 
            argname: str, 
            study, 
    ):
        x = np.load(f'arch/{argname}/study_metrics/{name_a}').flatten()
        y = np.load(f'arch/{argname}/study_metrics/{name_b}').flatten()
        z = np.load(f'arch/{argname}/study_metrics/{name_c}').flatten()

        if "AUC" in name_a and "Loss" in name_b and "Size" in name_c:
            op_x = "max"
            op_y = "min"
            op_z = "min"
        elif "AUC" in name_b and "Loss" in name_a and "Size" in name_c:
            op_x = "min"
            op_y = "max"
            op_z = "min"
        data = pd.DataFrame({
            name_a: x, 
            name_b: y, 
            name_c: z, 
        })
        mask = paretoset(data, sense=[op_x, op_y, op_z])
        pareto_data = data[mask]

        x_pareto=pareto_data.get(name_a).to_numpy().flatten()
        y_pareto=pareto_data.get(name_b).to_numpy().flatten()
        z_pareto=pareto_data.get(name_c).to_numpy().flatten()
        pareto_ind = np.argsort(x_pareto)
        x_pareto=x_pareto[pareto_ind]
        y_pareto=y_pareto[pareto_ind]
        z_pareto=z_pareto[pareto_ind]
        
        trials_list = [trial.params for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.RUNNING]
        trials_list = [params for params, keep in zip(trials_list, mask) if keep]
        trials_list = [trials_list[i] for i in pareto_ind.tolist()]
        
        return [x_pareto, y_pareto, z_pareto, trials_list]
    
    def get_3d_pareto_lite(
            self, 
            name_a: str, 
            name_b: str, 
            name_c: str, 
            argnames: list, 
            studies: list, 
    ):
        x, y, z = [], [], []
        for i, argname in enumerate(argnames):
            x.append(np.load(f'arch/{argname}/study_metrics/{name_a}').flatten())
            y.append(np.load(f'arch/{argname}/study_metrics/{name_b}').flatten())
            z.append(np.load(f'arch/{argname}/study_metrics/{name_c}').flatten())
        x = np.concatenate(x)
        y = np.concatenate(y)
        z = np.concatenate(z)

        if "AUC" in name_a and "Loss" in name_b and "Size" in name_c:
            op_x = "max"
            op_y = "min"
            op_z = "min"
        elif "AUC" in name_b and "Loss" in name_a and "Size" in name_c:
            op_x = "min"
            op_y = "max"
            op_z = "min"
        data = pd.DataFrame({
            name_a: x, 
            name_b: y, 
            name_c: z, 
        })
        mask = paretoset(data, sense=[op_x, op_y, op_z])
        pareto_data = data[mask]

        x_pareto=pareto_data.get(name_a).to_numpy().flatten()
        y_pareto=pareto_data.get(name_b).to_numpy().flatten()
        z_pareto=pareto_data.get(name_c).to_numpy().flatten()
        pareto_ind = np.argsort(x_pareto)
        x_pareto=x_pareto[pareto_ind]
        y_pareto=y_pareto[pareto_ind]
        z_pareto=z_pareto[pareto_ind]
        
        trials_list = []
        for i in range(len(studies)):
            trials_list_temp = [trial.params for trial in studies[i].trials if trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.RUNNING]
            trials_list = trials_list + trials_list_temp
        trials_list = [params for params, keep in zip(trials_list, mask) if keep]
        trials_list = [trials_list[i] for i in pareto_ind.tolist()]
        
        return [x_pareto, y_pareto, z_pareto, trials_list]
    
    def get_3d_pareto_executions(
            self, 
            argname, 
            study, 
            name_x: str, 
            name_y: str, 
            trial_names: List, 
            num_trials_in_study: list = [],
    ):
        if type(argname) == str:
            size = np.load(f'arch/{argname}/study_metrics/Model Size (b).npy')
        elif type(argname == list):
            size = []
            for i in range(len(argname)):
                size.append(np.load(f'arch/{argname[i]}/study_metrics/Model Size (b).npy'))
            size = np.concatenate(size)
        name_z = 'Model Size (b)'

        x = np.array([])
        y = np.array([])
        z = np.array([])
        if type(argname) == str:
            trials = [trial.params for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.RUNNING]
            trials_list_all = []
            for i in range(len(trial_names)):
                x_new = np.load(f'arch/{argname}/trial_metrics/{name_x}/{trial_names[i]}').flatten()
                y_new = np.load(f'arch/{argname}/trial_metrics/{name_y}/{trial_names[i]}').flatten()
                x = np.append(x, x_new)
                y = np.append(y, y_new)
                z = np.append(z, np.ones(x_new.shape[-1]) * size[i])
                for j in range(x_new.shape[-1]):
                    trials_list_all.append(trials[i])
        elif type(argname) == list:
            trials = [trial.params for i in range(len(argname)) for trial in study[i].trials if trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.RUNNING]
            trials_list_all = []
            num_trials_in_study_diff = np.cumsum(num_trials_in_study)
            num_trials_in_study_diff = np.concatenate((np.array([0]), num_trials_in_study_diff))
            for i in range(len(argname)):
                for j in range(num_trials_in_study[i]):
                    x_new = np.load(f'arch/{argname[i]}/trial_metrics/{name_x}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten()
                    y_new = np.load(f'arch/{argname[i]}/trial_metrics/{name_y}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten()
                    z_new = np.load(f'arch/{argname[i]}/trial_metrics/{name_z}/{trial_names[j + num_trials_in_study_diff[i]]}').flatten()
                    for k in range(x_new.shape[-1]):
                        trials_list_all.append(trials[j + num_trials_in_study_diff[i]])

        if "AUC" in name_x and "Loss" in name_y:
            op_x = "max"
            op_y = "min"
            op_z = "min"
        elif "AUC" in name_y and "Loss" in name_x:
            op_x = "min"
            op_y = "max"
            op_z = "min"
        
        pareto_set = []
        data = pd.DataFrame({
            name_x: x, 
            name_y: y, 
            name_z: z,  
        })
        mask = paretoset(data, sense=[op_x, op_y, op_z])
        pareto_data = data[mask]
        x_pareto=pareto_data.get(name_x).to_numpy().flatten()
        y_pareto=pareto_data.get(name_y).to_numpy().flatten()
        z_pareto=pareto_data.get(name_z).to_numpy().flatten()
        pareto_ind = np.argsort(x_pareto)
        x_pareto=x_pareto[pareto_ind]
        y_pareto=y_pareto[pareto_ind]
        z_pareto=z_pareto[pareto_ind]
        
        trials_list = [params for params, keep in zip(trials_list_all, mask) if keep]
        trials_list = [trials_list[i] for i in pareto_ind.tolist()]
        pareto_set.append([x_pareto, y_pareto, z_pareto, trials_list])

        return pareto_set
