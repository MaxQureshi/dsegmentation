#!/bin/bash
cd src
FILES=../experiments/*.param
for f in $FILES;
do
filename=$(basename "$f");
echo ${filename};

while IFS='' read -r arguments || [[ -n "$arguments" ]];
do 
echo "python3 train.py $arguments"
python3 train.py $arguments
done < "../experiments/${filename}"
done

