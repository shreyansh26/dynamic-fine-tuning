MODEL_PATH="/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-14T23:11:17_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_sft/epoch_1/step_final"
# MODEL_PATH="/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-15T00:08:26_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_dft/epoch_1/step_final"

vllm serve $MODEL_PATH \
    --port 8835 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --chat-template ${MODEL_PATH}/chat_template.jinja