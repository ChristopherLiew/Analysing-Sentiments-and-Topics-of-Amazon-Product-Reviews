import typer
import models.sent_clf.hf_clf_cli as hf_sent_clf

app = typer.Typer()
# Dataset preparation
# Huggingface Sentiment Clf
app.add_typer(hf_sent_clf.app, name='hf-sent-clf')
# SKLearn Sentiment Clf

@app.command()
def load():
    """
    Awesome portal gun
    """
    typer.echo("Loading rounds")
    

@app.command()
def shoot(name: str):
    """
    Shoot gun at Name.

    Args:\n
        name (str): Target of choice
    """
    typer.echo(f"Pew {name}!")
