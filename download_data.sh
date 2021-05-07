#!/usr/bin/env bash
# prepare dir
rm -rf data/*
mkdir -p data

# download data
wget -O data/cmc.csv https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data
wget -O data/wine.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
wget -O data/occupancy.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip
wget -O data/diabetes.csv http://nrvis.com/data/mldata/pima-indians-diabetes.csv
wget -O data/bank.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
wget -O data/skin.txt https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt

# prepare data
sed -i '1d' data/wine.csv
sed -i 's/;/,/g' data/wine.csv

unzip data/occupancy.zip -d data/
sed -i '1d' data/datatest.txt data/datatest2.txt data/datatraining.txt
cat data/datatest.txt data/datatest2.txt data/datatraining.txt >> data/occupancy.csv
rm data/datatest.txt data/datatest2.txt data/datatraining.txt data/occupancy.zip
cut -d, -f1-2 --complement data/occupancy.csv >> data/occupancy_temp.csv
mv data/occupancy_temp.csv data/occupancy.csv

unzip data/bank.zip -d data/
rm data/bank-names.txt data/bank.csv data/bank.zip
mv data/bank-full.csv data/bank.csv
sed -i '1d' data/bank.csv
sed -i 's/;/,/g' data/bank.csv
sed -i 's/"//g' data/bank.csv

sed -i 's/\t/,/g' data/skin.txt
sed -i 's/ /,/g' data/skin.txt
mv data/skin.txt data/skin.csv
