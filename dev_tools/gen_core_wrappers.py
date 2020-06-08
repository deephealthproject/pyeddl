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
