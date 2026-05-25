from pathlib import Path
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


MODEL_DIR = Path("models/logit")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_model_struct(X_dense, y, n_features, n_classes):
    with pm.Model() as logistic_model:
        # ---PRIORS---
        intercept = pm.Normal("intercept", mu=0, sigma=1, shape=n_classes)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(n_features, n_classes))

        # --- LINEAR MODEL ---
        raw_scores = intercept + pm.math.dot(X_dense, betas)
        p = pm.Deterministic("probs", pm.math.softmax(raw_scores, axis=1))

        # --- LIKELIHOOD ---
        obs = pm.Categorical("obs", p=p, observed=y)

    return logistic_model


# TRAIN
def train_log_model(
    X_train,
    y_train,
    y_classes,
    ppc_samples=100,
    n_iter=1000,
    n_iter_burn=1000,
    n_chains=4,
):
    X_train_dense = X_train.toarray()
    n_samples, n_features = X_train_dense.shape
    n_classes = len(y_classes)

    logistic_model = build_model_struct(
        X_dense=X_train_dense, y=y_train, n_features=n_features, n_classes=n_classes
    )

    with logistic_model:
        print("Started Logistic Model MCMC sampling")
        idata = pm.sample(
            draws=n_iter, tune=n_iter_burn, chains=n_chains, cores=4, random_seed=123
        )

        print("Saving Logistic Model MCMC samples")
        az.to_netcdf(idata, MODEL_DIR / "trace.nc")

        draws_per_chain = max(1, ppc_samples // n_chains)

        idata_ppc_sample = idata.isel(draw=slice(0, draws_per_chain))
        post_pred_train = pm.sample_posterior_predictive(
            idata_ppc_sample,
            var_names=["probs", "obs"],
        )

        print("Saving posterior train samples")
        az.to_netcdf(post_pred_train, MODEL_DIR / "posterior_train.nc")

    return idata, post_pred_train


# TEST
def test_log_model(X_test, y_classes, y_test=None):
    idata = az.from_netcdf(MODEL_DIR / "trace.nc")

    X_test_dense = X_test.toarray()
    n_samples, n_features = X_test_dense.shape
    n_classes = len(y_classes)
    if y_test is not None:
        y_obs = y_test
    else:
        y_obs = np.zeros(n_samples, dtype=int)

    logistic_model = build_model_struct(
        X_dense=X_test_dense, y=y_obs, n_features=n_features, n_classes=n_classes
    )

    with logistic_model:
        # PREDIKCE
        post_pred_test = pm.sample_posterior_predictive(
            idata,
            model=logistic_model,
            var_names=["probs", "obs"],
        )
        # TEST PPC
        if y_test is not None:
            az.plot_ppc(post_pred_test, num_pp_samples=150, kind="kde")
            plt.savefig("figures/logit/ppc_test.png")
            plt.close()

    # (chains, draws, n_samples, n_classes) -> prumer pres chains a draws -> vysledny tvar bude (n_samples, n_classes) -> sklearn
    probs = (
        post_pred_test.posterior_predictive["probs"].mean(dim=["chain", "draw"]).values
    )

    return probs

