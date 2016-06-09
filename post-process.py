#!/usr/bin/env python3
# post-processing dirty OCR for C19 french newspapers project
# assumes a series of .txt files in a 'to_clean' directory at the
# same level as this script.

# import modules
import os
from os import remove
import re
import codecs
from shutil import move


# the name of the directories containing the dirty OCR'd text
CORPUS = 'to_clean'

#
patterns = [
            (r'111+', 'll'),
            (r'mére', 'mère'),
            (r'pére', 'père'),
            (r'c\b', 'e'),
            (r'ct', 'et'),
            (r'cn', 'en'),
            (r'\.\.\b', '«'),
            (r'\b\.\.', '»')
]


def all_file_names(dirname=CORPUS):
    """Reads in the files"""
    for (root, _, files) in os.walk(dirname):
        for fn in files:
            yield os.path.join(root, fn)


def do_the_replacing(filenames):
    """takes the files in and replaces everything"""
    for file in filenames:
            with codecs.open(file, 'r+', 'utf8') as f:
                with codecs.open(file + '_temp', 'w', 'utf8') as new_f:
                    for line in f.readlines():
                        for pattern in patterns:
                            line = re.sub(*pattern, line)
                        line = re.sub('change', 'obama', line)
                        new_f.write(line)
            remove(file)
            move(file + '_temp', file)


def main():
    filenames = list(all_file_names())
    do_the_replacing(filenames)

if __name__ == '__main__':
    main()
