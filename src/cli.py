import typer
from src.scoring.score_latest import score_latest

app = typer.Typer()


@app.command()
def score_latest_cmd():
    """
    Run latest scoring pipeline.
    """
    score_latest()   # ← remove typer.echo


if __name__ == "__main__":
    app()