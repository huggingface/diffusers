# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse


def overwrite_file(file, class_name, test_name, correct_line):
    with open(file, "r") as f:
        lines = f.readlines()

    class_regex = f"class {class_name}("
    test_regex = f"{4 * ' '}def {test_name}("
    line_begin_regex = f"{8 * ' '}{correct_line.split()[0]}"
    in_class = False
    in_func = False
    in_line = False
    insert_line = False

    new_lines = []
    for line in lines:
        if line.startswith(class_regex):
            in_class = True
        elif in_class and line.startswith(test_regex):
            in_func = True
        elif in_class and in_func and line.startswith(line_begin_regex):
            in_line = True

        if in_class and in_func and in_line:
            if ")" not in line:
                continue
            else:
                insert_line = True

        if in_class and in_func and in_line and insert_line:
            new_lines.append(f"{8 * ' '}{correct_line}")
            in_class = in_func = in_line = insert_line = False
        else:
            new_lines.append(line)

    with open(file, "w") as f:
        for line in new_lines:
            f.write(line)


def main(fail, correct):
    with open(fail, "r") as f:
        test_failures = set([l.strip() for l in f.readlines()])

    with open(correct, "r") as f:
        correct_lines = f.readlines()

    for line in correct_lines:
        file, class_name, test_name, correct_line = line.split(";")
        if "::".join([file, class_name, test_name]) in test_failures:
            overwrite_file(file, class_name, test_name, correct_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail_filename", help="filename of test failures")
    parser.add_argument("--correct_filename", help="filename of tests with expected result")
    args = parser.parse_args()

    main(args.fail_filename, args.correct_filename)
