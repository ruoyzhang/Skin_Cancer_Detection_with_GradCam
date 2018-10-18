#!/bin/sh

pip install kaggle
mkdir data
cd data/
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip
chmod 755 *
unzip HAM10000_images_part_1.zip -d images #/train
unzip HAM10000_images_part_2.zip -d images #/train
rm *.zip

# SEPERATING TEST & TRAIN
#cd images/train
#NB_FILE=$(find . -type f -name "*.jpg" | wc -l)
#NB_TEST=$((NB_FILE / 10))
#mkdir ../test
#echo 'Importing pictures to test folder...'
#for f in $(ls | gshuf -n $NB_TEST); do mv "$f" ../test; done