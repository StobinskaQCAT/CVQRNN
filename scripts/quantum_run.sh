#!/bin/bash

# Script for running quantum/classical.py. It is useful for running the
# program with few different parameters one after another. Naming of the
# parameters is consistent with the naming used in the python program.

# Specify parameters of the run or put them into the loop
DATA_NAME="bessel"
IN_LEN=4
OUT_LEN=1
IN_MODES=1
HID_MODES=2
CUTOFF=6
BATCH_SIZE=10
MAX_EPOCHS=50
GAMMA=10
LR=0.01
LAYERS=1
REPETITIONS=1
NONLINEAR=0
ENERGY_TRANSMISSIVITY=1
RUN_NUM=5
DATA_LEN=22
RESOLUTION=200
NOISE=0

# go to one directory above the script directory i.e. QRNN
cd "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")"/.. || exit

for DATA_NAME in "bessel"; do
  ALL_MODES=$(($IN_MODES + $HID_MODES))

  # Too big cutoff can use all the RAM
  #  if [ $HID_MODES == 1 ]; then
  #    CUTOFF=6
  #  elif [ $HID_MODES == 2 ]; then
  #    CUTOFF=6
  #  elif [ $HID_MODES == 3 ]; then
  #    CUTOFF=4
  #  fi

  # Create the name of the directory, where all the results will be saved
  SAVE_DIR="${DATA_NAME}_modes_${ALL_MODES}_cut_${CUTOFF}_lr_${LR}"
  SAVE_DIR+="_inlen_${IN_LEN}_outlen_${OUT_LEN}_datalen_${RESOLUTION}"
  SAVE_DIR+="_gamma_${GAMMA}_layers_${LAYERS}_nonlinear_${NONLINEAR}"
  SAVE_DIR+="_noise_${NOISE}_channel_loss_${ENERGY_TRANSMISSIVITY}"
  #  SAVE_DIR+="immediate_loss"

  # Create the directory if it does not exist yet
  mkdir -p "./figures/quantum_runs/${SAVE_DIR}"
  mkdir -p "./results/quantum"

  # Print all the parameters of the run into 'parameters.dat' file
  echo -e "Parameters of this run:

    DATA_NAME\t=\t${DATA_NAME}
    IN_LEN\t=\t${IN_LEN}
    OUT_LEN\t=\t${OUT_LEN}
    IN_MODES\t=\t${IN_MODES}
    HID_MODES\t=\t${HID_MODES}
    CUTOFF\t=\t${CUTOFF}
    BATCH_SIZE\t=\t${BATCH_SIZE}
    MAX_EPOCHS\t=\t${MAX_EPOCHS}
    GAMMA\t=\t${GAMMA}
    LR\t=\t${LR}
    LAYERS\t=\t${LAYERS}
    REPETITIONS\t=\t${REPETITIONS}
    NONLINEAR\t=\t${NONLINEAR}
    ENERGY_TRANSMISSIVITY\t=\t${ENERGY_TRANSMISSIVITY}
    RUN_NUM\t=\t${RUN_NUM}
    DATA_LEN\t=\t${DATA_LEN}
    RESOLUTION\t=\t${RESOLUTION}
    NOISE\t=\t${NOISE}

    COMMENTS:
      -> output process: average of the quad photons
      -> density matrix pass: partial
      -> loss: only on the last output L(x_T, y_T)
      -> encoding: Dgate
      -> channel loss = yes
    " >"./figures/quantum_runs/${SAVE_DIR}/parameters.dat"

  # Run the program with the specified parameters and including quantum module
  python3 -m QRNN.quantum.quantum \
    --data_name=$DATA_NAME \
    --in_len=$IN_LEN \
    --out_len=$OUT_LEN \
    --in_modes=$IN_MODES \
    --hid_modes=$HID_MODES \
    --cutoff=$CUTOFF \
    --batch_size=$BATCH_SIZE \
    --epochs=$MAX_EPOCHS \
    --gamma=$GAMMA \
    --lr=$LR \
    --layers=$LAYERS \
    --repetitions=$REPETITIONS \
    --nonlinear=$NONLINEAR \
    --energy_transmissivity=$ENERGY_TRANSMISSIVITY \
    --run_num=$RUN_NUM \
    --data_len=$DATA_LEN \
    --resolution=$RESOLUTION \
    --noise=$NOISE \
    --save_dir=$SAVE_DIR

done
