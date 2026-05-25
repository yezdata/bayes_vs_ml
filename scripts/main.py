from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import arviz as az

from scripts.services.logistic import test_log_model
from scripts.services.plots import (
    cm_plot,
    multiclass_roc_plot,
    forest_plot_n_features,
    mcmc_diag_plots,
)


# pro M4 GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Používám zařízení: {device}")


fig_path = Path("figures")
model_path = Path("models")
for d in [fig_path, model_path]:
    if not d.exists():
        d.mkdir()
models = ["naive_bayes", "logit", "finbert"]
for m in models:
    if not Path(fig_path / m).exists():
        Path(fig_path / m).mkdir()


df = pd.read_csv("data/all_data.csv")

X = df["Sentence"]
y = df["Sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

le = LabelEncoder()
le.fit(y_train)
for index, label in enumerate(le.classes_):
    print(f"{index} = {label}")
y_train = le.transform(y_train)
y_test = le.transform(y_test)


# UCENI TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf.fit(X_train)
joblib.dump(tfidf, model_path / "tfidf_vectorizer.pkl")
# tfidf = joblib.load(model_path / "tfidf_vectorizer.pkl")

X_train_vec = tfidf.transform(X_train)
X_test_vec = tfidf.transform(X_test)

feature_names = tfidf.get_feature_names_out()
df_readable = pd.DataFrame(X_train_vec.toarray(), columns=feature_names)

print(X_train_vec.shape)

# Naive Bayes
nb_model = ComplementNB()
nb_model.fit(X_train_vec, y_train)
nb_pred_class = nb_model.predict(X_test_vec)
nb_pred = nb_model.predict_proba(X_test_vec)


# Logistic Regression
# UCENI LOGIT
from scripts.services.logistic import train_log_model

idata, post_pred_train = train_log_model(
    X_train_vec,
    y_train,
    le.classes_,
    ppc_samples=150,
    n_iter=1000,
    n_iter_burn=1000,
    n_chains=4,
)

# idata = az.from_netcdf("models/logit/trace.nc")
# post_pred_train = az.from_netcdf("models/logit/posterior_train.nc")


# trace, autocorr, ppc train plots
mcmc_diag_plots(
    idata,
    post_pred_train,
    save_path=fig_path / "logit",
    n_features=15,
    n_pp_samples=150,
)

forest_plot_n_features(
    idata,
    tfidf,  # Váš natrénovaný TfidfVectorizer
    le,  # Váš LabelEncoder
    top_n=15,
    save_path=fig_path / "logit/forest.png",
)


summary_df = az.summary(idata)
rhat = az.rhat(idata)
ess = az.ess(idata)
print(f"Summary:\n{summary_df}")
print(f"Max R-hat: {rhat.max().to_array().max().values}")
print(f"Min Ess: {ess.min().to_array().min().values}")


logit_pred = test_log_model(X_test_vec, le.classes_, y_test=y_test)
logit_pred_class = logit_pred.argmax(axis=1)


# FinBERT
# UCENI FINBERT
from scripts.services.finbert import fine_tune_finbert

fine_tune_finbert(X_train=X_train, y_train=y_train)


model_path = "models/fin_bert_finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

pipe = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None
)

X_test_finb = X_test.astype(str).tolist()
finb_result = pipe(X_test_finb, batch_size=16)

finb_pred_df = pd.DataFrame(
    [{item["label"]: item["score"] for item in row} for row in finb_result]
)
finb_pred = finb_pred_df[le.classes_].values
finb_pred_class = le.transform(finb_pred_df.idxmax(axis=1))


# Evaluation
def evaluate(y_pred, y_pred_class, model_label, save_name):
    print(f"-----{model_label} Evaluation-----")
    print(classification_report(y_test, y_pred_class, target_names=le.classes_))
    # NOTE ROC AUC SCORE ZDE JE OVO!
    print(
        f"{model_label} Roc Auc Score: {roc_auc_score(y_test, y_pred, multi_class='ovo', average='weighted')}"
    )
    print(f"{model_label} Log loss: {log_loss(y_test, y_pred)}")

    cm_plot(
        y_test=y_test,
        y_pred_class=y_pred_class,
        model_label=model_label,
        y_classes=le.classes_,
        save_path=fig_path / save_name / "confusion_matrix.png",
    )
    # NOTE ROC AUC SCORE ZDE JE OVR!
    multiclass_roc_plot(
        y_test,
        y_pred,
        le.classes_,
        model_label,
        save_path=fig_path / save_name / "roc_curve.png",
    )


evaluate(nb_pred, nb_pred_class, "Naive Bayes", "naive_bayes")
evaluate(logit_pred, logit_pred_class, "Logistic Regression", "logit")
evaluate(finb_pred, finb_pred_class, "FinBERT", "finbert")
