export TARGET_FOLDER="/local/crv/acanberk/llama/weights/"
export MODEL_SIZE="13B"
export CUDA_VISIBLE_DEVICES=5,6,7 


#sets NPROC to appropriate model size
if [ $MODEL_SIZE == "7B" ]; then
    export LLAMA_NPROC=1
elif [ $MODEL_SIZE == "13B" ]; then
    export LLAMA_NPROC=2
elif [ $MODEL_SIZE == "33B" ]; then
    export LLAMA_NPROC=4
elif [ $MODEL_SIZE == "65B" ]; then
    export LLAMA_NPROC=8
else
    echo "Model size is not valid. Please use 7B, 13B, 33B, or 65B"
    exit 1
fi

# launch app.py and example.py in different processes
torchrun --nproc_per_node $LLAMA_NPROC example.py \
    --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE \
    --tokenizer_path $TARGET_FOLDER/tokenizer.model  2>&1 | tee /tmp/output.log &

while ! grep -q "Loaded in" /tmp/output.log; do
  sleep 0.1
done

rm /tmp/output.log

python app.py

pkill -P $$

