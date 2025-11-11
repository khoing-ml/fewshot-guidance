from fire import Fire

from .cli import main as cli_main


if __name__ == "__main__":
    Fire(
        {
            "t2i": cli_main,
            "control": control_main,
            "fill": fill_main,
            "kontext": kontext_main,
            "redux": redux_main,
        }
    )