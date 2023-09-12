#!/bin/bash
lr="3e-4"

for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03
do
	python -u main.py \
	--batch_size 64 \
	--n_epochs 20 \
	--data $data \
	--morph \
	--model "maveriq/morphgpt-base-100k" \
	--noise test \
	--output $1 \
	--lr $lr \
	--device 0
done
