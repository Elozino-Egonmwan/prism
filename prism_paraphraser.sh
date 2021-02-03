#!/usr/bin/env bash

SENTS="$1"

#Check that arguments have been passed in
: ${1:?'Requires sentence(s) to be paraphrased! Exiting...'}

#if environment name is not passed set to default
if [ -n "$2" ]; then
  ENVNAME=$2
else
  ENVNAME=prismenv
fi

#create conda env if it doesn't exist else activate it
ENVS=$(conda env list)
if [[ $ENVS == *$ENVNAME* ]]; then
   conda activate $ENVNAME
else
   conda create -n $ENVNAME python=3.7 -y
   conda activate $ENVNAME
fi;

#cd if necessary

#download model for the first time otherwise export only
MODEL_DIR="m39v1"
if [ -d $MODEL_DIR ]; then
  export MODEL_DIR=m39v1/
else
  echo "Downloading model"
  wget http://data.statmt.org/prism/m39v1.tar
  tar xf m39v1.tar
  export MODEL_DIR=m39v1/
fi;

#check if requirements are already installed else install
if ! [ -x "$(command -v fairseq-preprocess)" ]; then
  pip install -r requirements.txt
fi

#create file for paraphrase generation
echo "$SENTS"
python create_input_file.py --sents "${SENTS}"

#remove test_bin if it exists
if [ -d "test_bin" ]; then
  rm -Rf "test_bin"
fi

#create test bin with sents to be paraphrased
#fairseq-preprocess --source-lang src --target-lang tgt  \
#    --joined-dictionary  --srcdict $MODEL_DIR/dict.tgt.txt \
#    --trainpref  test  --validpref test  --testpref test --destdir test_bin

#generate paraphrases
#python paraphrase_generation/generate_paraphrases.py test_bin --batch-size 8 \
#   --prefix-size 1 \
#   --path $MODEL_DIR/checkpoint.pt \
#   --prism_a 0.003 --prism_b 4
