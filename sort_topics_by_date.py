import argparse
import csv
import sys


def parse_args(argv=None):
    """This parses the command line."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-f', '--file', dest='filename', action='store',
                        help='The input directory containing the training '
                             'corpus.')

    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        
    except:
        print(filename + " was not a csv file")
    rows = []


if __name__ == '__main__':
    main()
