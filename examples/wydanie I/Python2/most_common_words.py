# -*- coding: utf-8 -*-
# most_common_words.py
import sys
from collections import Counter

if __name__ == "__main__":

    # Jako pierwszy argument przekaż listę słów.
    try:
        num_words = int(sys.argv[1])
    except:
        print "Uruchomiono: most_common_words.py num_words"
        sys.exit(1)   # Kod wyjściowy inny niż zero świadczy o błędzie.

    counter = Counter(word.lower()                      
                      for line in sys.stdin             
                      for word in line.strip().split()  
                      if word)                          
            
    for word, count in counter.most_common(num_words):
        sys.stdout.write(str(count))
        sys.stdout.write("\t")
        sys.stdout.write(word)
        sys.stdout.write("\n")