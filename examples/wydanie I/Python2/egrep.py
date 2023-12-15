# -*- coding: utf-8 -*-
# egrep.py
import sys, re

if __name__ == "__main__":

    # sys.argv to lista argumentów wiersza poleceń.
    # sys.argv[0] to nazwa samego programu.
    # sys.argv[1] będzie wyrażeniem regularnym określonym w wierszu poleceń.
    regex = sys.argv[1]

    # Wykonaj dla każdej linii tekstu przekazanej do skryptu.
    for line in sys.stdin:
        # Zapisz do strumienia stdout, jeżeli pasuje do wyrażenia regularnego.
        if re.search(regex, line):
            sys.stdout.write(line)