import sys

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        print("featureguard requires Python 3.10 or newer.", file=sys.stderr)
        raise SystemExit(2)
    from .cli import main

    raise SystemExit(main())
