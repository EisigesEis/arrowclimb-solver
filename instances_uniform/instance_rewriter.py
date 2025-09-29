"""
Rewrite instance files of the form

X
m_1,s_1
...
Y
n_1,p_1
...

Replace line X with count of lines between X and Y,
and line Y with count of lines after Y.

Features:
- Rewrite a single file (output to stdout, -o OUTPUT, or --inplace)
- Rewrite all files in a folder with --inplace on a directory

Usage:
  python instance_rewriter.py INPUT [-o OUTPUT] [--inplace]

Examples:
# Rewrite all files in ./instances in place
    python instance_rewriter.py ./instances --inplace

# Rewrite a single file in place
    python instance_rewriter.py instance1.txt --inplace

# Rewrite a single file to another output file
    python instance_rewriter.py instance1.txt -o rewritten.txt
"""

import argparse
import re
import sys
from pathlib import Path

INT_LINE_RE = re.compile(r"^\s*\d+\s*$")

def find_headers(lines):
    idx_x = None
    idx_y = None
    for i, line in enumerate(lines):
        if INT_LINE_RE.match(line):
            if idx_x is None:
                idx_x = i
            else:
                idx_y = i
                break
    if idx_x is None or idx_y is None or idx_y <= idx_x:
        raise ValueError("Could not locate two standalone integer header lines (X then Y) in order.")
    return idx_x, idx_y

def rewrite_counts(text):
    lines = text.splitlines(keepends=True)
    idx_x, idx_y = find_headers(lines)

    def non_empty_count(slice_):
        return sum(1 for ln in slice_ if ln.strip() != "")

    machines = non_empty_count(lines[idx_x + 1 : idx_y])
    jobs = non_empty_count(lines[idx_y + 1 :])

    def line_ending_of(line):
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        if line.endswith("\r"):
            return "\r"
        return "\n"

    x_end = line_ending_of(lines[idx_x])
    y_end = line_ending_of(lines[idx_y])

    lines[idx_x] = f"{machines}{x_end}"
    lines[idx_y] = f"{jobs}{y_end}"

    return "".join(lines), machines, jobs

def process_file(path: Path, inplace=False, output: Path=None):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return False

    try:
        rewritten, machines, jobs = rewrite_counts(text)
    except ValueError as ve:
        print(f"Format error in {path}: {ve}", file=sys.stderr)
        return False

    if inplace:
        try:
            path.write_text(rewritten, encoding="utf-8", newline="")
        except Exception as e:
            print(f"Error writing {path}: {e}", file=sys.stderr)
            return False
        # print(f"Updated {path} (machines={machines}, jobs={jobs})")
        return True
    elif output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rewritten, encoding="utf-8", newline="")
        except Exception as e:
            print(f"Error writing {output}: {e}", file=sys.stderr)
            return False
        # print(f"Wrote {output} (machines={machines}, jobs={jobs})")
        return True
    else:
        sys.stdout.write(rewritten)
        return True

def main():
    ap = argparse.ArgumentParser(description="Rewrite X and Y header lines with line counts.")
    ap.add_argument("input", type=Path, help="Input file or folder path.")
    ap.add_argument("-o", "--output", type=Path, help="Optional output file path (only for single file input).")
    ap.add_argument("--inplace", action="store_true", help="Rewrite the input file(s) in place.")
    args = ap.parse_args()

    if args.input.is_dir():
        if not args.inplace:
            ap.error("When input is a directory, --inplace must be specified.")
        for file in args.input.iterdir():
            if file.is_file():
                process_file(file, inplace=True)
    else:
        if args.inplace and args.output:
            ap.error("Use either --inplace or --output, not both.")
        process_file(args.input, inplace=args.inplace, output=args.output)

if __name__ == "__main__":
    main()