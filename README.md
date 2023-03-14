
# Serve Multi-GPU LlaMa  on Flask!

This is a hacky script that simultaneously runs LLaMa and a web server so that you can launch a local LLaMa API. 

So far it supports running the 13B model on 2 GPUs but it can be extended to serving bigger models as well.

To get it running, just edit the top of the `launch.sh` script to input your `CUDA_VISIBLE_DEVICES`, `TARGET_FOLDER`, and `MODEL_SIZE`. Then, run `./launch.sh` and you should be good to go!

Feel free to improve this script, fork your own repo, or whatever you want!

# Original repository [and Readme] at https://github.com/facebookresearch/llama

# Changes From Previous Fork
- New *example.py* allows to use console for interactive prompting. Supports multiple gpu (tested with 13b model on two RTX3090)
Supports multiple gpu's (tested with 13b model on two RTX3090).
- Modified **llama/generate.py** to support the above functionality
- Batch size set to 1 (equivalent to no batches at all)
- The rest of the code is left unchanged
- Added option to split model 7B into two GPUs (just use option -MP=2 as for model 13B). Further parallelization is possible, but I don't plan to implement it.

