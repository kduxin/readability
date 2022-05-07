#! /bin/bash

SECTIONS="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"

for s in $SECTIONS; do
  echo $s
  cat /data/texts/english/penn/original/COMBINED/WSJ/${s}/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | gzip > train/words/train.${s}.words.gz
  cat /data/texts/english/penn/original/COMBINED/WSJ/${s}/* | wsj-removetraces.pl | wsj-to-se.pl -w 0 -p 1 | gzip > train/words/train.${s}.synt.wsj.gz
done
