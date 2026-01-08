"""CLI utilities for CoTLab."""

import shutil
from pathlib import Path

import click


@click.group()
def cli():
    """CoTLab command-line utilities."""
    pass


@cli.command()
@click.argument("model_name")
@click.option(
    "--backend",
    default="vllm",
    type=click.Choice(["vllm", "transformers"]),
    help="Backend: vllm or transformers",
)
@click.option("--output", "-o", help="Output path (default: conf/model/<safe_name>.yaml)")
def template(model_name: str, backend: str, output: str):
    """Generate model config from template.

    Examples:

        cotlab-template meta-llama/Llama-3.1-8B

        cotlab-template google/gemma-3-12b --backend transformers

        cotlab-template mistralai/Mistral-7B-v0.1 -o conf/model/mistral7b.yaml
    """
    # Get project root (where this file is located)
    cli_file = Path(__file__)
    project_root = cli_file.parent.parent.parent

    # Get template
    template_path = project_root / f"conf/model/_base/{backend}_default.yaml"

    if not template_path.exists():
        click.echo(f"Template not found: {template_path}", err=True)
        raise click.Abort()

    # Generate output path
    if not output:
        # Convert model name to safe filename
        # meta-llama/Llama-3.1-8B -> meta_llama_llama_3_1_8b
        safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
        output = f"conf/model/{safe_name}.yaml"

    output_path = Path(output)

    # Check if exists
    if output_path.exists():
        if not click.confirm(f"{output} already exists. Overwrite?"):
            click.echo("Aborted.")
            raise click.Abort()

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy template
    shutil.copy(template_path, output_path)

    # Replace placeholder
    content = output_path.read_text()
    content = content.replace("huggingface/model-name", model_name)
    output_path.write_text(content)

    # Success message
    click.echo(f"Created: {output}")
    click.echo(f"Edit {output} to customize parameters")
    click.echo("\nUsage:")
    click.echo(f"  python -m cotlab.main model={output_path.stem}")


if __name__ == "__main__":
    cli()
