# Continuous Variable Quantum Recurrent Neural Network

In this repository we gather code, with which we simulate our architecture
called CV-QRNN. The initial results are gathered in the preprint
[https://arxiv.org/abs/2207.00378](https://arxiv.org/abs/2207.00378) by
Michał Siemaszko, Thomas McDermott, Adam Buraczewski, Bertrand Le Saux and
Magdalena Stobińska.

The structure of this repository is following:

```bash
QRNN
├── figures
│   ├── classical_runs
│   ├── comparison
│   ├── quantum_runs
│   └── scripts
├── logs
├── QRNN
│   ├── classicals
│   │   ├── mnist.py
│   │   └── classical.py
│   ├── quantum
│   │   ├── mnist.py
│   │   └── quantum.py
│   └── utils
│       ├── dataLoader.py
│       ├── LSTMCell.py
│       ├── plot.py
│       └── QRNNCell.py
├── README.md
└── scripts
    ├── classical_run.sh
    ├── classical_run_mnist.sh
    ├── quantum_run.sh        
    └── quantum_run_mnist.sh
```

- ```figures``` stores all produced figures: during the training and
  for comparison
- ```logs``` contains logs of all the runs in the *Tensorboard* form
- ```QRNN\classical``` contain scripts which implement classical LSTM
- ```QRNN\quantum``` contain scripts which implement quantum RNN
- ```QRNN\utils``` contain sets of useful functions and classes used by
  QRNN and LSTM (including visualisation and implementations of networks)
- ```scripts``` contains scripts which run quantum/classical network with
  set hyperparameters and architecture

To run a experiment:

1. prepare python enviroment ```pip install -r requirements.txt```
2. run the script

```bash
cd QRNN
./scripts/quantum_run.sh
```