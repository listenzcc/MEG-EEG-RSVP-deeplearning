#!/usr/bin/env zsh

source ~/.zshrc
conda activate python3.10

echo -----------------------------
echo python env
which python
python --version

# script=pipeline/decoding.1.raw.to.epochs.py
# script=test1.py
script=test2.py

# mode=EEG
# python $script --subj S01 --mode $mode 
# python $script --subj S02 --mode $mode 
# python $script --subj S03 --mode $mode 
# python $script --subj S04 --mode $mode 
# python $script --subj S05 --mode $mode 
# python $script --subj S06 --mode $mode 
# python $script --subj S07 --mode $mode 
# python $script --subj S08 --mode $mode 
# python $script --subj S09 --mode $mode 
# python $script --subj S10 --mode $mode

mode=MEG
python $script --subj S01 --mode $mode --device 5
python $script --subj S02 --mode $mode --device 5
python $script --subj S03 --mode $mode --device 5
python $script --subj S04 --mode $mode --device 5
python $script --subj S05 --mode $mode --device 5
python $script --subj S06 --mode $mode --device 5
python $script --subj S07 --mode $mode --device 5
python $script --subj S08 --mode $mode --device 5
python $script --subj S09 --mode $mode --device 5
python $script --subj S10 --mode $mode --device 5
