#!/usr/bin/env bash

WDIR=.
TRNSL_ENIS_SPM_MODEL=$WDIR/base_model/sentence.bpe.model
TRNSL_ENIS_DICT=$WDIR/dicts/dict.txt
TRNSL_ENIS_DATA=$DATA_BASE/bin

ENG=en_XX
ISL=is_IS
LANGS="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,is_IS,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"

#trnsl_enis_binarize_bi () {
#    local INPUT_FILE_SRC=$1
#    local INPUT_FILE_TGT=$2
#    local OUTPUT_PREFIX=$3
#    local TEMP_DIR=$(mktemp -d)
#    spm_encode --model="$TRNSL_ENIS_SPM_MODEL" --input="$INPUT_FILE_SRC" --output="$TEMP_DIR/text.$ENG"
#    spm_encode --model="$TRNSL_ENIS_SPM_MODEL" --input="$INPUT_FILE_TGT" --output="$TEMP_DIR/text.$ISL"
#    fairseq-preprocess \
#        --source-lang $ENG \
#        --target-lang $ISL \
#        --trainpref "$TEMP_DIR/text" \
#        --destdir "$TEMP_DIR" \
#        --srcdict "$TRNSL_ENIS_DICT" \
#        --joined-dictionary \
#        --workers 8
#    mkdir -p $TRNSL_ENIS_DATA
#    for ext in bin idx ; do
#        mv "$TEMP_DIR/train.$ENG-$ISL.$ENG.$ext" "$TRNSL_ENIS_DATA/$OUTPUT_PREFIX.$ENG-$ISL.$ENG.$ext"
#        mv "$TEMP_DIR/train.$ENG-$ISL.$ISL.$ext" "$TRNSL_ENIS_DATA/$OUTPUT_PREFIX.$ENG-$ISL.$ISL.$ext"
#    done
#    if [ ! -f "$TRNSL_ENIS_DATA/dict.txt" ] ; then
#        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.txt"
#        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.$ENG.txt"
#        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.$ISL.txt"
#    fi
#    rm -r "$TEMP_DIR"
#}
#
#mkdir -p $WDIR/outputs
#
#set -e
#
##################################################
##################################################
##############  binarize function  ###############
#
#trnsl_enis_binarize_bi () {
#    local INPUT_FILE_SRC=$1
#    local INPUT_FILE_TGT=$2
#    local OUTPUT_PREFIX=$3
#    local TEMP_DIR=$(mktemp -d)
#    spm_encode --model="$TRNSL_ENIS_SPM_MODEL" --input="$INPUT_FILE_SRC" --output="$TEMP_DIR/text.$ENG"
#    spm_encode --model="$TRNSL_ENIS_SPM_MODEL" --input="$INPUT_FILE_TGT" --output="$TEMP_DIR/text.$ISL"
#    fairseq-preprocess \
#        --source-lang $ENG \
#        --target-lang $ISL \
#        --trainpref "$TEMP_DIR/text" \
#        --destdir "$TEMP_DIR" \
#        --srcdict "$TRNSL_ENIS_DICT" \
#        --joined-dictionary \
#        --workers 8
#    mkdir -p $TRNSL_ENIS_DATA
#    for ext in bin idx ; do
#        mv "$TEMP_DIR/train.$ENG-$ISL.$ENG.$ext" "$TRNSL_ENIS_DATA/$OUTPUT_PREFIX.$ENG-$ISL.$ENG.$ext"
#        mv "$TEMP_DIR/train.$ENG-$ISL.$ISL.$ext" "$TRNSL_ENIS_DATA/$OUTPUT_PREFIX.$ENG-$ISL.$ISL.$ext"
#    done
#    if [ ! -f "$TRNSL_ENIS_DATA/dict.txt" ] ; then
#        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.txt"
#        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.$ENG.txt"
#        cp "$TRNSL_ENIS_DICT" "$TRNSL_ENIS_DATA/dict.$ISL.txt"
#    fi
#    rm -r "$TEMP_DIR"
#}

#################################################
#################################################
#############  evaluate on flores dev  ##########

FLORES_SRC=$WDIR/flores101_dataset/dev/eng.dev
FLORES_REF=$WDIR/flores101_dataset/dev/isl.dev
TMP_DOMAINS_FILE=$WDIR/data/domains.unspec

#cp $FLORES_SRC $TMP_DOMAINS_FILE
#sed -i 's/.*$/domain1/' $TMP_DOMAINS_FILE
#
#TRNSL_ENIS_DATA=$WDIR/data/bin
#trnsl_enis_binarize_bi \
#    $FLORES_SRC \
#    $FLORES_REF \
#    valid.dev.flores

CKPT=$WDIR/checkpoints/debug/checkpoint_last.pt
fairseq-generate $WDIR/data/bin/ \
        --path $CKPT \
        --langs $LANGS \
        --no-progress-bar \
        --task translation_from_pretrained_bart_domain \
        --domain-dict $WDIR/data/raw/domains.txt \
        --bpe 'sentencepiece' --sentencepiece-model /data/models/mbart25-cont-enis/sentence.bpe.model \
        -s "$ENG" -t "$ISL" \
        --max-tokens 8000 \
        --user-dir $WDIR/fairseq_user_dir \
        --train-subset train --valid-subset valid.dev.flores \
        --train-domains $WDIR/data/raw/train.domains --valid-domains $TMP_DOMAINS_FILE \
        --gen-subset valid.dev.flores \
        --beam 4 --lenpen 1.0 --num-workers 4 \
        > $WDIR/flores.dev.output.txt

