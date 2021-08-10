import typer
from sent_clf_lib.fasttext import ft_embeds
from sent_clf_lib.huggingface import hf_clf
from sent_clf_lib.sklearn import rf_clf, svc_clf, text2embed

app = typer.Typer()

# Dataset preparation
# Take in raw data, preprocess it and then create a data dir

# Train fasttext embeddings
app.add_typer(ft_embeds.app, name="ft-embeds", help="Train fast text word embeddings")

# Convert preprocessed tokenized text into embeddings
app.add_typer(
    text2embed.app,
    name="text2embed",
    help="""Converts text 2 embeddings and vectorises or pools
                  them for sklearn models""",
)

# Transformer
app.add_typer(hf_clf.app, name="hf-clf", help="Huggingface sequence classifier.")

# Random Forest Classifier
app.add_typer(rf_clf.app, name="rf-clf", help="Random Forest sequence classifier.")

# Support Vector Classifier
app.add_typer(svc_clf.app, name="svc-clf", help="Support vector sequence classifier.")
