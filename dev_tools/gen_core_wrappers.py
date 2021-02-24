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
Generate pure python wrappers from pybind11 function bindings.

Does not do the whole job, but saves most of the typing.
"""

import argparse
import io
import os
import re
import string


FUNC_PATTERN = re.compile(r'(\S+)\.def\("(\w+)"')
ARGS_PATTERN = re.compile(r'pybind11::arg\("(\w+)"\)')
KWARGS_PATTERN = re.compile(r'pybind11::arg\("(\w+)"\) *= *([^,)]+)')
MODULE_PATTERN = re.compile(r'M\("(\w+)"\)')

TEMPLATE = string.Template("""\
def ${func_name}(${signature}):
    return _${module}.${func_name}(${all_args})
""")


def parse(line):
    m = FUNC_PATTERN.search(line)
    if not m:
        return None, None, None, None
    if "cl.def" in line:  # skip method defs
        return None, None, None, None
    module, func_name = m.groups()
    if module == "m":
        module = None
    else:
        try:
            module = MODULE_PATTERN.match(module).groups()[0]
        except AttributeError:
            print("module: %r" % (module,))
            raise
    kwargs = KWARGS_PATTERN.findall(line)
    keys = set(_[0] for _ in kwargs)
    args = [_ for _ in ARGS_PATTERN.findall(line) if _ not in keys]
    return module, func_name, args, kwargs


def gen_out_func(module, func_name, args, kwargs):
    signature = ""
    if args:
        signature += ", ".join(args)
        if kwargs:
            signature += ", "
    if kwargs:
        signature += ", ".join(["%s=%s" % (k, v) for k, v in kwargs])
    all_args = ", ".join(args + [_[0] for _ in kwargs])
    return TEMPLATE.substitute(
        func_name=func_name,
        signature=signature,
        module=module,
        all_args=all_args
    )


def main(args):
    if not args.out_file:
        head, tail = os.path.splitext(args.in_file)
        args.out_file = "%s_wrappers.py" % head
    print("reading from %s and writing to %s" % (args.in_file, args.out_file))
    with io.open(args.in_file, "rt") as f, io.open(args.out_file, "wt") as fo:
        for line in f:
            line = line.strip()
            module, func_name, args_, kwargs = parse(line)
            if func_name is None:
                continue
            if module is None:
                module = args.module
            fo.write("\n\n")
            fo.write(gen_out_func(module, func_name, args_, kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_file", metavar="INPUT_FILE",
                        help="input pybind11 C++ file")
    parser.add_argument("-o", "--out-file", metavar="FILE", help="output file")
    parser.add_argument("-m", "--module", help="pyeddl submodule name",
                        default="eddl")
    main(parser.parse_args())
