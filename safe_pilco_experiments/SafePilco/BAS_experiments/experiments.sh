#!/bin/bash
for i in {0..9}
do
python bas_safePILCO.py 1 $i $1 # $1 is the path to BASBEnchmarks source folder, eg: /Users/XXXX/BASBenchmarks/src
done
