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

"""\
Find methods that have both a static and a non-static version in the
auto-generated bindings (not supported by pybind11, since it's not allowed in
python). Work around the issue by replacing the static binding from
def_static("<NAME>", ...) to def_static("static_<NAME>", ...)
"""

import argparse
import io
import re
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).absolute().parent
BINDINGS_FILE = THIS_DIR.parent / "src" / "_core.cpp"

CLASS_DEF_PATTERN = re.compile(r"pybind11::class_<(\w+)")
METH_DEF_PATTERN = re.compile(r'cl\.def\("(\w+)"')
STATIC_METH_DEF_PATTERN = re.compile(r'cl\.def_static\("(\w+)"')


def get_defs(f):
    defs_by_class = {}
    klass = None
    for line in f:
        m_klass = CLASS_DEF_PATTERN.search(line.strip())
        m_meth = METH_DEF_PATTERN.search(line.strip())
        m_static_meth = STATIC_METH_DEF_PATTERN.search(line.strip())
        if m_klass:
            klass = m_klass.groups()[0]
            if klass == "std":  # last entry, commented out
                break
            defs_by_class[klass] = {"def": set(), "def_static": set()}
        elif m_meth:
            meth = m_meth.groups()[0]
            defs_by_class[klass]["def"].add(meth)
        elif m_static_meth:
            static_meth = m_static_meth.groups()[0]
            defs_by_class[klass]["def_static"].add(static_meth)
    return defs_by_class


def get_common_defs(defs_by_class):
    common_defs = {}
    for k, v in defs_by_class.items():
        common_defs[k] = v["def"] & v["def_static"]
    return common_defs


def replace_static_names(fin, fout, common_defs):
    klass = None
    meth_names = set()
    for line in fin:
        m_klass = CLASS_DEF_PATTERN.search(line.strip())
        m_static_meth = STATIC_METH_DEF_PATTERN.search(line.strip())
        if m_klass:
            klass = m_klass.groups()[0]
            if klass == "std":  # last entry, commented out
                meth_names = set()
            else:
                meth_names = common_defs[klass]
        elif m_static_meth and meth_names:
            static_meth = m_static_meth.groups()[0]
            if static_meth in meth_names:
                line = line.replace(
                    f'"{static_meth}"', f'"static_{static_meth}"'
                )
        fout.write(line)


def main(args):
    with io.open(BINDINGS_FILE, "rt") as f:
        defs = get_defs(f)
    common_defs = get_common_defs(defs)
    if args.dry_run:
        print("the following names are used for static and non-static methods")
        for k, v in sorted(common_defs.items()):
            if v:
                print(k)
                for name in sorted(v):
                    print(f"  {name}")
        return 0
    with io.open(BINDINGS_FILE, "rt") as fin:
        with tempfile.NamedTemporaryFile(mode="wt", delete=False) as fout:
            replace_static_names(fin, fout, common_defs)
    Path(fout.name).rename(args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out-file", metavar="FILE", help="output file",
                        default=BINDINGS_FILE)
    parser.add_argument("--dry-run", action="store_true",
                        help="just list the problematic method names and exit")
    main(parser.parse_args())
