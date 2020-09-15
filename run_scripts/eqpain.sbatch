#!/usr/bin/env bash
#SBATCH --mem  6GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "rivendell|belegost|shire|gondor|shelob|balrog|smaug|khazadum"
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user sbroome@kth.se
#SBATCH --output /Midgard/home/%u/projects/painface-recognition/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/projects/painface-recognition/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate pfr
cd ~/projects/painface-recognition
python main.py --config-file ${CONFIG_FILE} --test-run ${TEST_RUN} --subjects-overview metadata/horse_subjects.csv --train-subjects ${TRAIN_SUBJECTS} --val-subjects ${VAL_SUBJECTS} --test-subjects ${TEST_SUBJECTS} --job-identifier ${SLURM_JOB_ID}
