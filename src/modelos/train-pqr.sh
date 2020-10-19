# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"

#mkdir -p "${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/exp"
#mkdir -p "${TRAIN_LOGDIR}"

NUM_ITERATIONS=2000
python3 "${WORK_DIR}/train.py" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --dataset="pqr" \
  --dataset_dir="${DATASET}" \
  --decoder_output_stride=8 \
  --fine_tune_batch_norm=false \
  --logtostderr \
  --model_variant="xception_65" \
  --optimizer="momentum" \
  --output_stride=8 \
  --train_batch_size=4 \
  --train_crop_size=600,600 \
  --train_logdir="${TRAIN_LOGDIR}" \
  --train_split="train" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
