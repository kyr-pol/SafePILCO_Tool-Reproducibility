#!/bin/bash
for i in {1..10}
do
python swimmer.py results/swimmer/ $i
done
