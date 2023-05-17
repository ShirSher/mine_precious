# README.md
# An example of a run:​

# bsub -q gpu-short -app nvidia-gpu -env LSB_CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3 -gpu num=4:j_exclusive=no -R "select[mem>8000] rusage[mem=8000]" -o out.%J -e err.%J python3 \
# Adam, {lr}, batch_size, 
# mine_run.py 2 0.0003 500 200 1 2 combined 3 2 2 512 6 same
# ---

# sys.argv gets 4 values:
# [1] - the optimizer - one of three (1) - SGD,(2) - Adam, (3) - RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - train true/false 1/0
# [6] - Net num - 1-3 #set equal 3 Ignore
# [7] - traject/MNIST/combined - What form of mine to compute # combined Ignore
# [8] - number_descending_blocks - i.e. how many steps should the fc networks take,
#                               starting at 2048 nodes and every step divide the size 
#                                by 2. max number is 7                                
# [9] - number_repeating_blocks - the number of times to repeat a particular layer
# [10] - repeating_blockd_size - the size of nodes of the layer to repeat
# [11] - traject_max_depths
# [12] - traject_num_layers
# [13] - same/different minist/trajectory - to use the same image that the 
#        trajectory ran on as the joint distribution. # Ignore? 