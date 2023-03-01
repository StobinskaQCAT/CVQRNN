#!/bin/bash

# Script for running classical/classical.py. It is useful for running
# the program with few different parameters one after another. Naming
# of the parameters is consistent with the naming used in the python
# program.

## Number of parameters

#Here is quick summary about different networks and the number of
#trainable parameters for different number of units
#
#| units |  QRNN |  LSTM  |  GRU  |  RNN  |
#|-------|-------|--------|-------|-------|
#|  1    |   8   |   14   |   14  |    4  |
#|  2    |   14  |   35   |   33  |    9  |
#|  3    |   27  |   64   |   58  |   16  |
#|  4    |   44  |   101  |   89  |   25  |
#|  5    |   65  |    -   |   -   |   36  |

# Specify parameters of the run or comment them and put into loop
# DATA_NAME="sunspots"
IN_LEN=4
OUT_LEN=1
BATCH_SIZE=10
MAX_EPOCHS=200
UNITS=4
LR=0.01
RUN_NUM=5
DATA_LEN=22
RESOLUTION=200
NOISE=0

# go to one directory above the script directory i.e. QRNN
cd "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")"/.. || exit

for DATA_NAME in "bessel"; do

  # Create the name of the directory, where all the results
  # will be saved (we are in QRNN)
  SAVE_DIR="${DATA_NAME}_units_${UNITS}_lr_${LR}"
  SAVE_DIR+="_inlen_${IN_LEN}_outlen_${OUT_LEN}"
  SAVE_DIR+="_datalen_${RESOLUTION}_noise_${NOISE}_LSTM"

  # Create the directory if it does not exist yet
  mkdir -p "./figures/classical_runs/${SAVE_DIR}"
  mkdir -p "./results/classical"

  # Print all the parameters of the run into 'parameters.dat' file
  echo -e "Parameters of this run:

    DATA_NAME\t=\t${DATA_NAME}
    UNITS\t=\t${UNITS}
    IN_LEN\t=\t${IN_LEN}
    OUT_LEN\t=\t${OUT_LEN}
    BATCH_SIZE\t=\t${BATCH_SIZE}
    MAX_EPOCHS\t=\t${MAX_EPOCHS}
    LR\t=\t${LR}
    RUN_NUM\t=\t${RUN_NUM}
    DATA_LEN\t=\t${DATA_LEN}
    RESOLUTION\t=\t${RESOLUTION}
    NOISE\t=\t${NOISE}

    COMMENTS:
    " >"./figures/classical_runs/${SAVE_DIR}/parameters.dat"

  # Run the program with the specified parameters
  python3 -m QRNN.classical.classical \
    --data_name=$DATA_NAME \
    --in_len=$IN_LEN \
    --out_len=$OUT_LEN \
    --units=$UNITS \
    --batch_size=$BATCH_SIZE \
    --epochs=$MAX_EPOCHS \
    --lr=$LR \
    --run_num=$RUN_NUM \
    --data_len=$DATA_LEN \
    --resolution=$RESOLUTION \
    --noise=$NOISE \
    --save_dir=$SAVE_DIR

done
