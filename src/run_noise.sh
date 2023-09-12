#!/bin/bash
lr="3e-5"

for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03
do
	python3 -u main.py \
	--batch_size 32 \
	--n_epochs 20 \
	--data $data \
	--base \
	--noise test \
	--model "gpt2" \
	--lr $lr \
	--device 0
done

for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03
do
	python3 -u main.py \
	--batch_size 32 \
	--n_epochs 20 \
	--data $data \
	--morph \
	--noise test \
	--model "maveriq/morphgpt-base-200k" \
	--lr $lr \
	--device 0
done


for k in 1 2 3 4
do 
	for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03
	do
		python -u main.py \
		--batch_size 32 \
		--n_epochs 20 \
		--data $data \
		--flota \
		--k $k \
		--noise test \
		--model "gpt2" \
		--lr $lr \
		--device 0
	done
done
