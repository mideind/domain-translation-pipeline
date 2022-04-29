#!/usr/bin/env bash
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

set -ex

WDIR=.
MODIFIED_CHECKPOINT=$WDIR/modified_restore_checkpoint.pt

LANGS="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,is_IS,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"

NVECS=58
LEARNING_RATE=5e-05
UPDATE_FREQ=3
EXPERIMENT=debug

TMP_DOMAINS_FILE=$WDIR/data/domains.unspec
fairseq-train $WDIR/data/bin \
    --no-progress-bar \
    --domain-dict $WDIR/data/raw/domains.txt \
    --encoder-normalize-before --decoder-normalize-before \
    --user-dir $WDIR/fairseq_user_dir \
    --arch mbart_large --layernorm-embedding \
    --task translation_from_pretrained_bart_domain \
    --source-lang en_XX --target-lang is_IS \
    --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr-scheduler reduce_lr_on_plateau --lr ${LEARNING_RATE} --best-checkpoint-metric ppl --warmup-updates 100 \
    --train-subset train --valid-subset valid.dev.flores --num-workers 8 \
    --train-domains $WDIR/data/raw/train.domains --valid-domains $TMP_DOMAINS_FILE \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
    --patience 16 \
    --max-tokens 2000 --update-freq ${UPDATE_FREQ} \
    --seed 222 --log-format simple --log-interval 10 \
    --validate-interval-updates 2000 \
    --keep-interval-updates 4 \
    --langs $LANGS \
    --save-dir ${WDIR}/checkpoints/$EXPERIMENT \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --restore-file $MODIFIED_CHECKPOINT --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \


