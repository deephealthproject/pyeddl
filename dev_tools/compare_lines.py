# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
