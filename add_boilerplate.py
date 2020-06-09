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
