for seed in 123; do
    CUDA_VISIBLE_DEVICES=2 nohup python -u train_PPO.py --normalize 0 --save-model --env metaworld_door-open-v2 --seed $seed --lr 0.0005 --batch-size 32 --n-envs 8 --ent-coef 0.0 --n-steps 250 --total-timesteps 500000 --num-layer 3 --hidden-dim 128 --clip-init 0.2 --gae-lambda 0.95 > raw_logs/dooropen_$seed.log 2>& 1 &
done
