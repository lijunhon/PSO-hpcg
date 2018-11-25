#!/bin/bash
#file == last
file=`ls *.yaml |sort >> result_file.txt && sed -n '$p' result_file.txt`
#gflop.txt == gflop
gflop=`cat $file |grep 'HPCG result' |cut -d ':' -f 2 |cut -d ' ' -f 2`
echo $gflop
