#!/bin/bash

data="arxiv_physics_1e+03"

for lr in 1e-4 1e-5 3e-5
do
	python -u main.py \
	--batch_size 64 \
	--n_epochs 20 \
	--data $data \
	--morph \
	--model "maveriq/morphgpt-base-200k" \
	--lr $lr \
	--device 0 \
    --hs
done