#!/usr/bin/env bash
#SBATCH --mem  4GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "rivendell|shire|gondor|shelob|balrog|smaug|khazadum|belegost"
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user sbroome@kth.se
#SBATCH --output /Midgard/home/%u/projects/painface-recognition/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/projects/painface-recognition/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate pfr
cd ~/projects/painface-recognition/1d-pain/
python main.py --config-file ${CONFIG_FILE} --nb-pain-train ${NB_PAIN_TRAIN} --nb-nopain-train ${NB_NOPAIN_TRAIN} --nb-pain-val ${NB_PAIN_VAL} --nb-nopain-val ${NB_NOPAIN_VAL} --job-identifier ${SLURM_JOB_ID}
