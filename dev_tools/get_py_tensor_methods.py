import io
import re
from pathlib import Path


THIS_DIR = Path(__file__).absolute().parent
FN = THIS_DIR.parent / "pyeddl" / "tensor.py"

PATTERN = re.compile(r"def (\w+)\(")


def get_methods():
    static, dynamic = set(), set()
    is_static = False
    with io.open(FN, "rt") as f:
        for line in f:
            if line.lstrip().startswith("@staticmethod"):
                is_static = True
                continue
            m = PATTERN.search(line)
            if not m:
                continue
            meth = m.groups()[0]
            if is_static:
                static.add(meth)
                is_static = False
            else:
                dynamic.add(meth)
    return static, dynamic


def main():
    static, dynamic = get_methods()
    with open("static.txt", "wt") as fs, open("dynamic.txt", "wt") as fd:
        for meth in static:
            fs.write(f"{meth}\n")
        for meth in dynamic:
            fd.write(f"{meth}\n")


if __name__ == "__main__":
    main()
