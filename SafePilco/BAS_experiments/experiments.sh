#!/bin/bash
cd BASBenchmarks/src/safe_pilco/
for i in {1..10}
do
python bas_safePILCO.py 1 $i
#python smart_buildings.py 1 $i
done
