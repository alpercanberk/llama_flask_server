#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./launch.sh MODEL_SIZE PORT"
    exit 1
fi

export TARGET_FOLDER="/local/crv/acanberk/llama/weights/"
export MODEL_SIZE=$1  # Set MODEL_SIZE to the first argument passed to the script
export PORT=$2  # Set PORT to the second argument passed to the script

# Sets NPROC to appropriate model size
case $MODEL_SIZE in
    "7B")
        export LLAMA_NPROC=1
        ;;
    "13B")
        export LLAMA_NPROC=2
        ;;
    "30B")
        export LLAMA_NPROC=4
        ;;
    "65B")
        export LLAMA_NPROC=8
        ;;
    *)
        echo "Invalid model size"
        exit 1
        ;;
esac

# Launch example.py in a different process
torchrun --nproc_per_node $LLAMA_NPROC example.py \
    --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE \
    --tokenizer_path $TARGET_FOLDER/tokenizer.model  2>&1 | tee /tmp/output.log &

while ! grep -q "Loaded in" /tmp/output.log; do
  sleep 0.1
done

rm /tmp/output.log

# Launch app.py with the specified port
python app.py --port $PORT

pkill -P $$
