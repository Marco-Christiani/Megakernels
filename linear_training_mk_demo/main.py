import runpy
from pathlib import Path

def main():
    runpy.run_path(
        Path(__file__).parent.joinpath("tests", "test_example.py").as_posix(),
        run_name="__main__",
    )

if __name__ == "__main__":
    main()