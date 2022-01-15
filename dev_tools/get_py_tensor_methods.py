# Copyright (c) 2019-2022 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
