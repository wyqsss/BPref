
for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
for step in 252000 302000 352000 402000 452000;do
CUDA_VISIBLE_DEVICES=2 python eval_ppo.py --seed $seed --video-path p2video --visual --test-epochs 1 --env metaworld_door-open-v2 --reload-path checkpoints/metaworld_door-open-v2/seed-$seed-step-$step.zip
done
done