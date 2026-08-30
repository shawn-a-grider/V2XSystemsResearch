#!/bin/bash

mkdir -p data

curl "https://data.transportation.gov/resource/iq8k-ytf6.csv?\$limit=500000" \
  -o data/spat_sample500k.csv

  