"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """YOLOv7 with SAHI."""
    print("Hello, world!")


if __name__ == "__main__":
    main(prog_name="yolov7-SAHI")  # pragma: no cover
