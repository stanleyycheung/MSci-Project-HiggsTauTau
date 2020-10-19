#!/bin/sh

# Starting CMSSW enviroment

echo "Starting CMSSW enviroment"

cd CMSSW_10_2_19/src

eval `scramv1 runtime -sh`

cd ..

cd ..

echo "CMSSW enviroment startup completed!"