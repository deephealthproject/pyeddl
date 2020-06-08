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
Extract Doxygen comments from a C++ header file, and convert them to Sphinx.
"""

import argparse


# quick and dirty
def doc_stream(f):
    buffer = []
    in_doc = False
    next_line = False
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("/**"):
            in_doc = True
            buffer.append('"""\\')
            continue
        if line.endswith("*/"):
            in_doc = False
            buffer.append('"""')
            continue
        if in_doc:
            line = line.lstrip("*").strip()
            if not line:
                continue
            if next_line:
                buffer.append("\n" + line + "\n")
                next_line = False
                continue
            try:
                k, v = line.split(None, 1)
            except ValueError:
                next_line = True
                continue
            if k == "@brief":
                buffer.append(v)
                continue
            if k == "@param":
                name, v = v.split(None, 1)
                k = f"{k} {name}"
            buffer.append(f":{k[1:]}: {v}")
        if buffer and buffer[-1] == '"""':
            yield line, "    " + "\n    ".join(buffer)
            buffer = []


def main(args):
    with open(args.in_file) as f, open(args.out_file, "w") as fo:
        for sig, doc in doc_stream(f):
            fo.write(f"{sig}\n")
            fo.write(f"{doc}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_file", metavar="INPUT_FILE",
                        help="input C++ source file")
    parser.add_argument("-o", "--out-file", metavar="FILE", help="output file",
                        default="docs.txt")
    main(parser.parse_args())
