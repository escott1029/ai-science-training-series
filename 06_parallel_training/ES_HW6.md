# 🚀 Parallel Training Methods for AI

[Sam Foreman](https://samforeman.me)  
[Intro to AI-driven Science on Supercomputers](https://www.alcf.anl.gov/alcf-ai-science-training-series)  
_2024-11-05_

- Slides: <https://samforeman.me/talks/ai-for-science-2024/slides>
  - HTML version: <https://samforeman.me/talks/ai-for-science-2024>

## 👋 Hands On

1. Submit interactive job:

    ```bash
    qsub -A ALCFAITP -q by-node -l select=1 -l walltime=01:00:00,filesystems=eagle:home -I
    ```

1. On Sophia:

    ```bash
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
    ```

1. Clone repos:

    1. [`saforem2/wordplay`](https://github.com/saforem2/wordplay):

        ```bash
        git clone https://github.com/saforem2/wordplay
        cd wordplay
        ```

    1. [`saforem2/ezpz`](https://github.com/saforem2/ezpz):

        ```bash
        git clone https://github.com/saforem2/ezpz deps/ezpz
        ```

1. Setup python:

    ```bash
    export PBS_O_WORKDIR=$(pwd) && source deps/ezpz/src/ezpz/bin/utils.sh
    ezpz_setup_python
    ezpz_setup_job
    ```

1. Install `{ezpz, wordplay}`:

    ```bash
    python3 -m pip install -e deps/ezpz --require-virtualenv
    python3 -m pip install -e . --require-virtualenv
    ```

1. Setup (or disable) [`wandb`](https://wandb.ai):

    ```bash
    # to setup:
    wandb login
    # to disable:
    export WANDB_DISABLED=1
    ```

1. Test Distributed Setup:

    ```bash
    mpirun -n "${NGPUS}" python3 -m ezpz.test_dist
    ```

    See: [`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)

1. Prepare Data:

    ```bash
    python3 data/shakespeare_char/prepare.py
    ```

1. Launch Training:

    ```bash
    mpirun -n "${NGPUS}" python3 -m wordplay \
        train.backend=DDP \
        train.eval_interval=100 \
        data=shakespeare \
        train.dtype=bf16 \
        model.batch_size=64 \
        model.block_size=1024 \
        train.max_iters=1000 \
        train.log_interval=10 \
        train.compile=false
    ```

## 🎒 Homework

Submit _proof_ that you were able to successfully follow the above instructions and launch a distributed data parallel training run.

Where _proof_ can be any of:

- The contents printed out to your terminal during the run
- A path to a logfile containing the output from a run on the ALCF filesystems
- A screenshot of:
  - the text printed out from the run
  - a graph from the W&B Run
  - anything that shows that you clearly were able to run the example
- url to a W&B Run or [W&B Report](https://api.wandb.ai/links/aurora_gpt/7du35js1)
- etc.



Terminal Contents:
```
What is an LLM?

DUKE VINCENTIO:
It is too green, to for with the lance of his guilty
post my complaining and my majesty. I'll believe him to your grace,
In an undertakent when you have done, if he request
he was combined to chasting him.

MARIANA:
Be ruled by him thence
[2024-11-27 06:47:36.039499][INFO][trainer.py:762] - Saving checkpoint to: /home/emilyscott/outputs/runs/pytorch/DDP/2024-11-27/06-35-30
[2024-11-27 06:47:36.040136][INFO][trainer.py:763] - Saving model to: /home/emilyscott/outputs/runs/pytorch/DDP/2024-11-27/06-35-30/model.pth
[2024-11-27 06:47:38.842279][INFO][configs.py:141] - Appending /home/emilyscott/outputs/runs/pytorch/DDP/2024-11-27/06-35-30 to /home/emilyscott/wordplay/src/ckpts/checkpoints.log
wandb: | 0.028 MB of 0.226 MB uploaded
wandb: Run history:
wandb:                      Loss/iter ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     Loss/lossf █▆▆▆▆▆▆▆▅▅▄▄▄▄▄▃▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                       Loss/mfu ▆▆▇▇▆▆▇▇▇▇▅▆▇▆▆▅▅▅▅▆▂▁▁▃▃▄▄▅▆▆▆▇▇██▇▇▇▇▄
wandb:                     Loss/train ████▅▅▅▅▄▄▄▄▃▃▃▃▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                       Loss/val ████▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▃▃▃▃▆▆▆▆▆▆▆▆▇▇▇▇▆▆▆▆
wandb:                  Timing/dt_avg ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█
wandb:                 Timing/dt_iter ▂▂▁▂▂▁▁▂▂▂▂▁▁▁▂▃▂▂▁▂█▄▂▁▂▁▂▁▁▂▂▁▁▁▁▂▁▂▁█
wandb:                  Timing/dt_tot ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█
wandb:                 Timing/dtb_avg ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█
wandb:                 Timing/dtb_tot ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█
wandb:                 Timing/dtf_avg ▂▂▂▂▁█▁▁▁▂▂▂▁▁▁▁▁▁▂▁▃▃▂▁▁▁▂▂▂▁▁▂█▂▂▁▁▁▃▁
wandb:                 Timing/dtf_tot ▂▂▂▂▁█▁▁▁▂▂▂▁▁▁▁▁▁▂▁▃▃▂▁▁▁▂▂▂▁▁▂█▂▂▁▁▁▃▁
wandb:                    Timing/iter ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:         Timing/samples_per_sec ▇▇█▇▆▇█▇▇▆▆██▇▇▅▇▆█▇▁▄▆▇▇▇▇▇█▇▇███▇▇▇▇▇▁
wandb: Timing/samples_per_sec_per_gpu ▇▇█▇▆▇█▇▇▆▆██▇▇▅▇▆█▇▁▄▆▇▇▇▇▇█▇▇███▇▇▇▇▇▁
wandb:            Timing/startup_time ▁
wandb:          Timing/tokens_per_sec ▇▇█▇▆▇█▇▇▆▆██▇▇▅▇▆█▇▁▄▆▇▇▇▇▇█▇▇███▇▇▇▇▇▁
wandb:  Timing/tokens_per_sec_per_gpu ▇▇█▇▆▇█▇▇▆▆██▇▇▅▇▆█▇▁▄▆▇▇▇▇▇█▇▇███▇▇▇▇▇▁
wandb:                  Training/iter ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  Training/loss █▆▆▆▆▆▆▆▅▅▄▄▄▄▄▃▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:              Training/loss_tot █▆▆▆▆▆▆▆▅▅▄▄▄▄▄▃▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                    Training/lr ▁▃▅▆████████████████████████████████████
wandb:                         dt_avg ▂▂▂▄▂▃▃▇▂▃▆▇██▂▄▁▂▂▆▆▂▆▃▂▃▂▃▂▃▁▄▁▄▂▁▁▂▄▃
wandb:                        dt_iter ▂▂▂▂▄▂▁▂▂▄▃▂▃▂▂▃▂▃▂▃█▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▂▁▂▂
wandb:                         dt_tot ▂▂▂▄▂▃▃▇▂▃▆▇██▂▄▁▂▂▆▆▂▆▃▂▃▂▃▂▃▁▄▁▄▂▁▁▂▄▃
wandb:                            dtb ▁▂▂▃▂▃▃▆▂▂▆▆█▂▂▄▁▂▂▆▅▂▅▂▁▃▁▂▂▂▁▄▁▃▂▂▁▂▃▃
wandb:                        dtb_avg ▁▂▂▃▂▃▃▆▂▂▆▆█▂▂▄▁▂▂▆▅▂▅▂▁▃▁▂▂▂▁▄▁▃▂▂▁▂▃▃
wandb:                        dtb_tot ▁▂▂▃▂▃▃▆▂▂▆▆█▂▂▄▁▂▂▆▅▂▅▂▁▃▁▂▂▂▁▄▁▃▂▂▁▂▃▃
wandb:                            dtf ▁▁▁▁▁▁▁▂▁▂▁▂▁█▁▁▁▁▁▁▂▁▁▂▂▁▁▁▁▁▁▂▁▂▁▁▁▁▂▁
wandb:                        dtf_avg ▁▁▁▁▁▁▁▂▁▂▁▂▁█▁▁▁▁▁▁▂▁▁▂▂▁▁▁▁▁▁▂▁▂▁▁▁▁▂▁
wandb:                        dtf_tot ▁▁▁▁▁▁▁▂▁▂▁▂▁█▁▁▁▁▁▁▂▁▁▂▂▁▁▁▁▁▁▂▁▂▁▁▁▁▂▁
wandb:                           iter ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           loss █▆▆▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▁▁▁▁▁▁▁▂▁▁▂▁▁▁▁▁▁▁
wandb:                       loss_tot █▆▆▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▁▁▁▁▁▁▁▂▁▁▂▁▁▁▁▁▁▁
wandb:                             lr ▁▃▅▆████████████████████████████████████
wandb:                samples_per_sec ▆▇▇▆▅▆▇▇▆▄▅▇▅▇▇▆▆▅▆▆▁▅▆▆▆▇▆▆▇▆▇▆▇█▆▆▇█▆▆
wandb:        samples_per_sec_per_gpu ▆▇▇▆▅▆▇▇▆▄▅▇▅▇▇▆▆▅▆▆▁▅▆▆▆▇▆▆▇▆▇▆▇█▆▆▇█▆▆
wandb:                 tokens_per_sec ▆▇▇▆▅▆▇▇▆▄▅▇▅▇▇▆▆▅▆▆▁▅▆▆▆▇▆▆▇▆▇▆▇█▆▆▇█▆▆
wandb:         tokens_per_sec_per_gpu ▆▇▇▆▅▆▇▇▆▄▅▇▅▇▇▆▆▅▆▆▁▅▆▆▆▇▆▆▇▆▇▆▇█▆▆▇█▆▆
wandb: 
wandb: Run summary:
wandb:                      Loss/iter 1000
wandb:                     Loss/lossf 0.03281
wandb:                       Loss/mfu 41.84364
wandb:                     Loss/train 0.04468
wandb:                       Loss/val 3.67581
wandb:                  Timing/dt_avg 0.13738
wandb:                 Timing/dt_iter 0.54293
wandb:                  Timing/dt_tot 0.27477
wandb:                 Timing/dtb_avg 0.26858
wandb:                 Timing/dtb_tot 0.26858
wandb:                 Timing/dtf_avg 0.00619
wandb:                 Timing/dtf_tot 0.00619
wandb:                    Timing/iter 999
wandb:         Timing/samples_per_sec 14.73491
wandb: Timing/samples_per_sec_per_gpu 1.84186
wandb:            Timing/startup_time 12.39993
wandb:          Timing/tokens_per_sec 965666.94349
wandb:  Timing/tokens_per_sec_per_gpu 120708.36794
wandb:                  Training/iter 999
wandb:                  Training/loss 0.03281
wandb:              Training/loss_tot 0.03281
wandb:                    Training/lr 0.0006
wandb:                         dt_avg 0.13738
wandb:                        dt_iter 0.54293
wandb:                         dt_tot 0.27477
wandb:                            dtb 0.26858
wandb:                        dtb_avg 0.26858
wandb:                        dtb_tot 0.26858
wandb:                            dtf 0.00619
wandb:                        dtf_avg 0.00619
wandb:                        dtf_tot 0.00619
wandb:                           iter 999
wandb:                           loss 0.03281
wandb:                       loss_tot 0.03281
wandb:                             lr 0.0006
wandb:                samples_per_sec 14.73491
wandb:        samples_per_sec_per_gpu 1.84186
wandb:                 tokens_per_sec 965666.94349
wandb:         tokens_per_sec_per_gpu 120708.36794
```



<!--[^gpu]: If you do not have access to the ALCF systems, you can install [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/) and run across multiple CPUs as well-->
