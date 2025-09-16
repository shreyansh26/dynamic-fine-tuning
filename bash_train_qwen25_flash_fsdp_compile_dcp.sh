set -x
torchrun --nnodes=1 --nproc-per-node=4 --master_port 29518 run_train_qwen25_fsdp.py --dcp-api --loss-type dft