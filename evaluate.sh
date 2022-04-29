#!/usr/bin/env bash
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

WDIR=.
TRNSL_ENIS_SPM_MODEL=$WDIR/base_model/sentence.bpe.model
TRNSL_ENIS_DICT=$WDIR/dicts/dict.txt
TRNSL_ENIS_DATA=$DATA_BASE/bin

ENG=en_XX
ISL=is_IS
LANGS="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,is_IS,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"

#################################################
#################################################
#############  evaluate on flores dev  ##########

FLORES_SRC=$WDIR/flores101_dataset/dev/eng.dev
FLORES_REF=$WDIR/flores101_dataset/dev/isl.dev
TMP_DOMAINS_FILE=$WDIR/data/domains.unspec

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

