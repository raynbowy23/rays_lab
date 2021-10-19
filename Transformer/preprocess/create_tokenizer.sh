#!/bin/sh
pipenv run spm_train --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --input=$HOME/InfoSci/OSS/Transformer/preprocess/natsume.txt --model_prefix=spm_natsume --vocab_size=8000