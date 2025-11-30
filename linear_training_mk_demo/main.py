import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).parent.joinpath("tests", "test_example.py").as_posix(),
        run_name="__main__",
    )
