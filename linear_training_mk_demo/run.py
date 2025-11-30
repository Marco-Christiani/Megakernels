from __future__ import annotations
import subprocess
import sys
import time
import signal
import selectors
from plumbum import local
from plumbum import colors

# Accuracy :)
# from plumbum.cmd import tput
# GREEN = tput("setaf", "2")
# BOLD = tput("bold")
# RESET = tput("sgr0")
# print(repr(GREEN))


def print_timer(start: float) -> None:
    elapsed = int(time.time() - start)
    mins, secs = divmod(elapsed, 60)
    # sys.stdout.write(f"\r{BOLD}{GREEN}Elapsed: {mins:02d}:{secs:02d}{RESET}")
    timer_str = colors.green & colors.bold | f"Elapsed: {mins:02d}:{secs:02d}"
    sys.stdout.write("\r" + timer_str)
    sys.stdout.flush()


def main() -> None:
    build = local["uv"]["pip", "install", "-e", ".", "--no-progress"]

    # non-blocking popen w/ selector
    process = build.popen(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    sel = selectors.DefaultSelector()
    sel.register(process.stdout, selectors.EVENT_READ)

    start = time.time()
    last_timer = 0.0

    def terminate(*_):
        try:
            process.terminate()
        except Exception:
            pass
        print()  # newline after timer
        sys.exit(1)

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    while True:
        now = time.time()
        if now - last_timer >= 0.25:  # every 0.25s
            print_timer(start)
            last_timer = now

        # poll for output w/o blocking
        events = sel.select(timeout=0)
        for key, _ in events:
            line = key.fileobj.readline()
            if line:
                # write normal output below timer
                sys.stdout.write("\n" + line.rstrip() + "\n")
                print_timer(start)

        if process.poll() is not None:
            break  # process ended

        time.sleep(0.05)

    print()
    sys.exit(process.wait())


if __name__ == "__main__":
    main()
