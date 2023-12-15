# -*- coding: utf-8 -*-
# line_count.py
import sys

if __name__ == "__main__":

    count = 0
    for line in sys.stdin:
        count += 1

    # Wynik wywołania funkcji print jest kierowany do strumienia sys.stdout.
    print count