import typer
from models.sent_clf import hf_clf

app = typer.Typer()
# Dataset preparation
# Take in raw data, preprocess it and then create a data dir

# Add in logging for dataset artifacts + test out models in full + inference
app.add_typer(hf_clf.app, 
              name='hf-clf',
              help='Huggingface sequence classifier.')

# SKLearn Sentiment Clf