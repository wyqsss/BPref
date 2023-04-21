for seed in 12345; do
    CUDA_VISIBLE_DEVICES=2 python train_PPO.py --env metaworld_push-v2 --seed $seed --lr 0.0003 --batch-size 128 --n-envs 16 --ent-coef 0.0 --n-steps 250 --total-timesteps 20000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92
done
