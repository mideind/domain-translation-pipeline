# Domain Translation Pipeline
A Domain Translation Pipeline built on Fairseq and mBART-derived translation models

This is a description of how to setup, prepare, train and then run the GreynirSeq
domain-aware translation model as part of the Icelandic LT Programme's machine translation core project.

This pipeline is typically set up using `conda`.
The requirements can be installed as follows (or alternatively, install `torch` and `fairseq` with `pip`):

    >>> conda env create -n domain_translation --file environment.yaml

Remember to activate your conda environment:

    >>> conda activate domain_translation

A valid checkpoint of a base translation model based on mBART25 can be finetuned as a domain translation model.
We recommend https://repository.clarin.is/repository/xmlui/handle/20.500.12537/125.
It is assumed to be located in the directory `./base_model`. An example command line to download it is:

    >>> mkdir -p base_model && pushd base_model && curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/125{/data-bin_enis.zip,/infer_enis.sh,/sentence.bpe.model,/mbart_nmt_isen.pt,/mbart_nmt_enis.pt,/infer_isen.sh} && popd

Relevant paths can be changed in the following files before they are executed:

- `./preprocess.sh`
- `./finetune.sh`
- `./evaluate.sh`

The `preprocess.sh` file expects a `.tsv` input file with the three fields (domains, english, icelandic), this is the training corpus.
By default the `preprocess.sh` script points to `./data/dummy.tsv` as the training corpus.

The order of execution is `preprocess.sh`, then `finetune.sh` until convergence,
and finally `evaluate.sh` to compute BLEU over `flores.dev`.
