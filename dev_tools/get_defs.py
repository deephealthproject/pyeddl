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
Extract function/method defs from a C++ header file.

Note that the approach is very naive, expecting a fairly 'regular' input. In
particular, it assumes that functions/methods are only declared, but not
implemented.
"""

import argparse
import io
import os
import re


def remove_comments(txt):
    # multiline comments
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=(re.MULTILINE | re.DOTALL))
    # inline comments
    txt = re.sub(r"//.*$", "", txt, flags=re.MULTILINE)
    return txt


def get_defs(txt, names_only=False):
    defs = []
    pattern = re.compile(r"(\w+) *\(")
    for line in txt.splitlines():
        m = pattern.search(line)
        if m:
            if names_only:
                defs.append(m.groups()[0])
            else:
                defs.append(line.lstrip())
    return "\n".join(defs)


def main(args):
    if not args.out_file:
        head, tail = os.path.splitext(args.in_file)
        args.out_file = "%s.defs" % head
    print("reading input file")
    with io.open(args.in_file, "rt") as f:
        txt = f.read()
    txt = remove_comments(txt)
    txt = get_defs(txt, names_only=args.names_only)
    print("writing to %s" % args.out_file)
    with io.open(args.out_file, "wt") as f:
        f.write(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_file", metavar="INPUT_FILE",
                        help="input C++ source file")
    parser.add_argument("-n", "--names-only", action="store_true",
                        help="output only function/method names")
    parser.add_argument("-o", "--out-file", metavar="FILE", help="output file")
    main(parser.parse_args())
