#!/bin/bash
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

set -e 
ENG=en_XX
ISL=is_IS

WDIR=.
INPUT_TSV_FILE=$WDIR/data/dummy.tsv
OUTPUT_DIR=$WDIR/data/raw
INPUT_CHECKPOINT=$WDIR/base_model/mbart_nmt_enis.pt
MODIFIED_CHECKPOINT=$WDIR/modified_restore_checkpoint.pt

DATA_BASE=$WDIR/data
mkdir -p $DATA_BASE/raw $DATA_BASE/bin

### split each domain into train, test and valid set
python3 preprocess.py $INPUT_TSV_FILE $OUTPUT_DIR

# download development dataset
if [ ! -d $WDIR/flores101_dataset ] ; then
    wget https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
    tar -xzf flores101_dataset.tar.gz
fi

TRNSL_ENIS_SPM_MODEL=$WDIR/base_model/sentence.bpe.model
TRNSL_ENIS_DICT=$WDIR/dicts/dict.txt
TRNSL_ENIS_DATA=$DATA_BASE/bin

trnsl_enis_binarize_bi () {
    local INPUT_FILE_SRC=$1
    local INPUT_FILE_TGT=$2
    local OUTPUT_PREFIX=$3
    local TEMP_DIR=$(mktemp -d)
    spm_encode --model="$TRNSL_ENIS_SPM_MODEL" --input="$INPUT_FILE_SRC" --output="$TEMP_DIR/text.$ENG"
    spm_encode --model="$TRNSL_ENIS_SPM_MODEL" --input="$INPUT_FILE_TGT" --output="$TEMP_DIR/text.$ISL"
    fairseq-preprocess \
        --source-lang $ENG \
        --target-lang $ISL \
        --trainpref "$TEMP_DIR/text" \
        --destdir "$TEMP_DIR" \
        --srcdict "$TRNSL_ENIS_DICT" \
        --joined-dictionary \
        --workers 8
    mkdir -p $TRNSL_ENIS_DATA
    for ext in bin idx ; do
        mv "$TEMP_DIR/train.$ENG-$ISL.$ENG.$ext" "$TRNSL_ENIS_DATA/$OUTPUT_PREFIX.$ENG-$ISL.$ENG.$ext"
        mv "$TEMP_DIR/train.$ENG-$ISL.$ISL.$ext" "$TRNSL_ENIS_DATA/$OUTPUT_PREFIX.$ENG-$ISL.$ISL.$ext"
    done
    if [ ! -f "$TRNSL_ENIS_DATA/dict.txt" ] ; then
        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.txt"
        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.$ENG.txt"
        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.$ISL.txt"
    fi
    rm -r "$TEMP_DIR"
}

TRNSL_ENIS_DATA=$DATA_BASE/bin/
trnsl_enis_binarize_bi \
    $DATA_BASE/raw/train.en \
    $DATA_BASE/raw/train.is \
    train

TRNSL_ENIS_DATA=$WDIR/data/bin
trnsl_enis_binarize_bi \
    $WDIR/flores101_dataset/dev/eng.dev \
    $WDIR/flores101_dataset/dev/isl.dev \
    valid.dev.flores

TMP_DOMAINS_FILE=$WDIR/data/domains.unspec
cp $WDIR/flores101_dataset/dev/eng.dev $TMP_DOMAINS_FILE
sed -i 's/.*$/domain1/' $TMP_DOMAINS_FILE

NDOMAINS=$(wc -l $OUTPUT_DIR/domains.txt | cut -f 1 -d' ')
echo "number of domains: '$NDOMAINS'"
python reshape_checkpoint_embeddings.py \
    --seed=1 \
    --input=$INPUT_CHECKPOINT \
    --output=$MODIFIED_CHECKPOINT \
    --nvecs=$NDOMAINS

