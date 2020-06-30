# Copyright (c) 2019-2020 CRS4
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
