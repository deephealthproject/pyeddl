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
Compare two text files and output lines that occur only in one of them.

Also prints info on lines that occur on both files, but with different
multiplicity.
"""

import argparse
import io
from collections import Counter


def get_lines(fname, strip=False):
    with io.open(fname, "rt") as f:
        gen = (_.rstrip() for _ in f) if strip else f
        return Counter(gen)


def main(args):
    c1 = get_lines(args.file_1, strip=args.strip)
    c2 = get_lines(args.file_2, strip=args.strip)
    print("Only in %s:" % args.file_1)
    for line in sorted(set(c1) - set(c2)):
        print("  %r" % (line,))
    print("Only in %s:" % args.file_2)
    for line in sorted(set(c2) - set(c1)):
        print("  %r" % (line,))
    print("different multiplicity:")
    for line in sorted(set(c1) & set(c2)):
        if c1[line] != c2[line]:
            print("  %r: %d, %d" % (line, c1[line], c2[line]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_1", metavar="FILE_1", help="fist input file")
    parser.add_argument("file_2", metavar="FILE_2", help="second input file")
    parser.add_argument(
        "-s", "--strip", action="store_true",
        help="strip leading and trailing whitespace before comparing"
    )
    main(parser.parse_args())
