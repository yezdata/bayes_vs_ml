from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import arviz as az
from itertools import cycle


def cm_plot(y_test, y_pred_class, model_label, y_classes, save_path=None):
    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=y_classes,
        yticklabels=y_classes,
    )
    plt.title(f"{model_label} Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# NOTE inverse transform tfidf a le a forest plot pro n nevyznamnejsich features v kazde tride
def forest_plot_n_features(idata, tfidf, le, top_n=10, save_path=None):
    posterior_mean = idata.posterior["betas"].mean(dim=["chain", "draw"]).values

    hdi = az.hdi(idata)["betas"].values

    feature_names = tfidf.get_feature_names_out()
    class_names = le.classes_

    colors = ["#d62728", "#7f7f7f", "#2ca02c"]

    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 8), sharey=False)

    if len(class_names) == 1:
        axes = [axes]

    for i, class_label in enumerate(class_names):
        ax = axes[i]

        class_betas = posterior_mean[:, i]
        class_hdi = hdi[:, i, :]

        sorted_indices = np.argsort(class_betas)[::-1][:top_n]

        top_words = feature_names[sorted_indices]
        top_means = class_betas[sorted_indices]
        top_hdis = class_hdi[sorted_indices]

        y_pos = np.arange(top_n)

        ax.scatter(top_means, y_pos, color=colors[i], s=100, zorder=3)

        x_err = np.array([top_means - top_hdis[:, 0], top_hdis[:, 1] - top_means])
        ax.errorbar(
            top_means,
            y_pos,
            xerr=x_err,
            fmt="none",
            ecolor=colors[i],
            alpha=0.6,
            capsize=5,
            zorder=2,
        )

        ax.axvline(0, color="black", linestyle="--", alpha=0.3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words, fontsize=12)
        ax.invert_yaxis()
        ax.set_title(
            f"Class: {class_label}", fontsize=14, color=colors[i], fontweight="bold"
        )
        ax.set_xlabel("Beta Coefficient")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def multiclass_roc_plot(y_test, y_score, classes, model_name, save_path=None):
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))

    colors = cycle(["blue", "red", "green"])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-class ROC: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def mcmc_diag_plots(idata, post_pred, save_path, n_features=15, n_pp_samples=100):
    # TRACE
    axes = az.plot_trace(idata, var_names=["intercept"])
    axes[0, 0].set_title(f"Overlay")
    axes[0, 1].set_title(f"Trace")
    plt.suptitle(f"Intercept", fontsize=16)
    plt.savefig(save_path / "trace_intercept.png")
    plt.close()

    # AUTOCORRELATION
    az.plot_autocorr(idata, var_names=["intercept"])
    plt.suptitle(f"Intercept - Autocorrelation", fontsize=26)
    plt.savefig(save_path / "autocorr_intercept.png")
    plt.close()

    for i in range(n_features):
        # TRACE
        axes = az.plot_trace(idata, var_names=["betas"], coords={"betas_dim_0": [i]})
        axes[0, 0].set_title(f"Overlay")
        axes[0, 1].set_title(f"Trace")
        plt.suptitle(f"Beta {i}", fontsize=16)
        plt.savefig(save_path / f"trace_beta_{i}.png")
        plt.close()

        # AUTOCORRELATION
        az.plot_autocorr(idata, var_names=["betas"], coords={"betas_dim_0": [i]})
        plt.suptitle(f"Beta {i} - Autocorrelation", fontsize=26)
        plt.savefig(save_path / f"autocorr_beta_{i}.png")
        plt.close()

    # PPC
    total_samples = (
        post_pred.posterior_predictive.sizes["chain"]
        * post_pred.posterior_predictive.sizes["draw"]
    )
    safe_samples = min(n_pp_samples, total_samples)

    az.plot_ppc(post_pred, num_pp_samples=safe_samples, kind="kde")
    plt.savefig(save_path / "ppc_train.png")
    plt.close()

