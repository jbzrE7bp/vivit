"""Plot the loss and accuracy over the training run for all configurations. Call this 
script via `python plot_loss_accuracy.py`.

NOTE: This script assumes that the training runs have already been performed and the 
`metrics.json`-files are present in the respective subfolders of `results/checkpoints`.
"""
import os

from config import CHECKPOINTS_OUTPUT, CONFIGURATIONS, PLOT_OUTPUT
from matplotlib import pyplot as plt
from plot_shared import get_config_title

from exp.utils.deepobs_runner import CheckpointRunner
from exp.utils.path import read_from_json

PLOT_SUBDIR = os.path.join(PLOT_OUTPUT, "loss_accuracy")


def get_plot_savedir():
    """Return sub-directory where results of the plotting script are saved."""
    savedir = PLOT_SUBDIR
    os.makedirs(savedir, exist_ok=True)
    return savedir


def get_plot_savepath(file_name, extension=".pdf"):
    """Get savepath for some result of the evaluation named `file_name`."""
    savedir = get_plot_savedir()
    return os.path.join(savedir, f"{file_name}{extension}")


if __name__ == "__main__":

    for config in CONFIGURATIONS:

        optimizer_cls = config["optimizer_cls"]
        problem_cls = config["problem_cls"]

        # ==============================================================================
        # Load metrics from json file
        # ==============================================================================
        savepath = CheckpointRunner.get_summary_savepath(
            problem_cls, optimizer_cls, CHECKPOINTS_OUTPUT
        )
        try:
            summary = read_from_json(savepath)
        except FileNotFoundError:
            print(f"Metrics not found at {savepath}. Skipping this config")
            continue

        train_losses = summary["train_losses"]
        test_losses = summary["test_losses"]
        train_acc_percent = [100 * val for val in summary["train_accuracies"]]
        test_acc_percent = [100 * val for val in summary["test_accuracies"]]

        num_epochs = config["num_epochs"]
        epochs = list(range(num_epochs + 1))

        # ==============================================================================
        # Generate the plot for this particular configuration
        # ==============================================================================

        config_title = get_config_title(config)
        col_train_loss = "#8a3829"
        col_test_loss = "#eb8775"
        col_train_acc = "#29708a"
        col_test_acc = "#71c5e3"

        # Loss
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="train", color=col_train_loss)
        ax.plot(epochs, test_losses, label="test", color=col_test_loss)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if "resnet32" in config_title:
            ax.set_yscale("log")
            ax.set_ylabel("loss (log scale)")

        # Additional settings
        ax.grid(ls="dashed", lw=0.4, dashes=(7, 7))
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_visible(True)
            ax.spines[axis].set_color("k")
            ax.spines[axis].set_linewidth(0.4)
        ax.xaxis.set_tick_params(width=0.5, direction="in", length=5)
        ax.yaxis.set_tick_params(width=0.5, direction="in", length=5)
        [t.set_color("k") for t in ax.xaxis.get_ticklabels()]
        [t.set_color("k") for t in ax.yaxis.get_ticklabels()]

        leg = ax.legend()
        leg.get_frame().set_linewidth(0.5)

        fig.savefig(get_plot_savepath(config_title + "_loss"))
        plt.close(fig)

        # Accuracy
        fig, ax = plt.subplots()
        ax.plot(epochs, train_acc_percent, label="train", color=col_train_acc)
        ax.plot(epochs, test_acc_percent, label="test", color=col_test_acc)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"accuracy in \%")

        # Additional settings
        ax.grid(ls="dashed", lw=0.4, dashes=(7, 7))
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_visible(True)
            ax.spines[axis].set_color("k")
            ax.spines[axis].set_linewidth(0.4)
        ax.xaxis.set_tick_params(width=0.5, direction="in", length=5)
        ax.yaxis.set_tick_params(width=0.5, direction="in", length=5)
        [t.set_color("k") for t in ax.xaxis.get_ticklabels()]
        [t.set_color("k") for t in ax.yaxis.get_ticklabels()]

        leg = ax.legend()
        leg.get_frame().set_linewidth(0.5)

        fig.savefig(get_plot_savepath(config_title + "_accuracy"))
        plt.close(fig)
