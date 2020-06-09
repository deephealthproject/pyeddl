# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
