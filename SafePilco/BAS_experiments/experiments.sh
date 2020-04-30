#!/bin/bash
cd BASBenchmarks/src/safe_pilco/
for i in {0..9}
do
python bas_safePILCO.py 1 $i
done
