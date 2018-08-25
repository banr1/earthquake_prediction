#!/bin/bash
python inference.py \
--model SimpleRNNmodel \
--epochs 20

python inference.py \
--model GRUmodel \
--epochs 40

python inference.py \
--model StackingGRUmodel \
--epochs 40

python inference.py \
--model LSTMmodel \
--epochs 40

python inference.py \
--model StackingLSTMmodel \
--epochs 40
