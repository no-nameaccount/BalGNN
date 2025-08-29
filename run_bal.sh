CUDA_VISIBLE_DEVICES=4 python main.py --dataset Cora --alpha 0.25 --beta 0.25 --model_type BalGNN --num_layers 8
CUDA_VISIBLE_DEVICES=4 python main.py --dataset CiteSeer --alpha 0.25 --beta 0.75 --model_type BalGNN --num_layers 8 
CUDA_VISIBLE_DEVICES=4 python main.py --dataset texas --alpha 0.75 --beta 0.25 --model_type BalGNN --num_layers 8 
CUDA_VISIBLE_DEVICES=4 python main.py --dataset cornell --alpha 0.25 --beta 0.25 --model_type BalGNN --num_layers 8 
CUDA_VISIBLE_DEVICES=4 python main.py --dataset wisconsin --alpha 0.5 --beta 1 --model_type BalGNN --num_layers 8
CUDA_VISIBLE_DEVICES=4 python main.py --dataset chameleon --alpha 0.5 --beta 1 --model_type BalGNN --num_layers 8 
