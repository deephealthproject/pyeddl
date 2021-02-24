# Copyright (c) 2019-2021 CRS4
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

"""\
Parse derived class definitions and output minimal bindings.

Naive approach, assumes a fairly regular input.
"""

import argparse
import io
import os
import re
from operator import itemgetter


PATTERN = re.compile(r"class\s*(\w+)\s*:\s*public\s*(\w+)\s*{")


def get_defs(fn):
    defs = set()
    print(f"reading {fn}")
    with io.open(fn, "rt") as f:
        for line in f:
            m = PATTERN.match(line)
            if m:
                defs.add(m.groups())
    return defs


def main(args):
    if not args.out_file:
        args.out_file = "bindings.cpp"
    defs = set()
    for root, dirs, files in os.walk(args.top_dir):
        for name in files:
            defs |= get_defs(os.path.join(root, name))
    with io.open(args.out_file, "wt") as f:
        for cls, parent in sorted(defs, key=itemgetter(1)):
            f.write(f'pybind11::class_<{cls}, std::shared_ptr<{cls}>, {parent}>(m, "{cls}", "");\n')  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("top_dir", metavar="TOP_DIR",
                        help="root of the dir tree with C++ header files")
    parser.add_argument("-o", "--out-file", metavar="FILE", help="output file")
    main(parser.parse_args())
