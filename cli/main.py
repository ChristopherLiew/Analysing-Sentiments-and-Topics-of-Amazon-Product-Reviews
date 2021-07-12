import typer
from models.sent_clf import hf_clf

app = typer.Typer()
# Dataset preparation
# Huggingface Sentiment Clf train + inference (predict)
app.add_typer(hf_clf.app, name='hf-sent-clf')
# SKLearn Sentiment Clf
