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

import io
import datetime
import os
import re


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIS_YEAR = datetime.date.today().year
C_YEAR = THIS_YEAR if THIS_YEAR == 2019 else f"2019-{THIS_YEAR}"
LICENSE_FN = os.path.join(THIS_DIR, "LICENSE")
COM_MAP = {
    ".py": "#",
    ".hpp": "//",
}
EXCLUDE_FILES = {
    "conf.py",
    "all_includes.hpp",
}
PATTERN = re.compile(r"Copyright \(c\) [0-9-]+")
REPL = f"Copyright (c) {C_YEAR}"


def get_boilerplate():
    with io.open(LICENSE_FN, "rt") as f:
        content = f.read()
    return content[content.find("Copyright"):]


def comment(text, com="#"):
    out_lines = []
    for line in text.splitlines():
        line = line.strip()
        out_lines.append(com if not line else f"{com} {line}")
    return "\n".join(out_lines) + "\n"


def add_boilerplate(boilerplate, fn):
    with io.open(fn, "rt") as f:
        text = f.read()
    if not text:
        return
    m = PATTERN.search(text)
    if m:
        # update existing
        with io.open(fn, "wt") as f:
            f.write(text.replace(m.group(), REPL))
        return
    # add new
    if text.startswith("#!"):
        head, tail = text.split("\n", 1)
        head += "\n\n"
    else:
        head, tail = "", text
    if not tail.startswith("\n"):
        boilerplate += "\n"
    with io.open(fn, "wt") as f:
        f.write(f"{head}{boilerplate}{tail}")


def main():
    join = os.path.join
    boilerplate = get_boilerplate()
    add_boilerplate(boilerplate, LICENSE_FN)
    bp_map = {ext: comment(boilerplate, com) for ext, com in COM_MAP.items()}
    for root, dirs, files in os.walk(THIS_DIR):
        dirs[:] = [_ for _ in dirs if not (
            _.startswith(".") or _ == "third_party"
        )]
        for name in files:
            if name in EXCLUDE_FILES:
                continue
            ext = os.path.splitext(name)[-1]
            try:
                bp = bp_map[ext]
            except KeyError:
                continue
            else:
                path = join(root, name)
                add_boilerplate(bp, path)


if __name__ == "__main__":
    main()
