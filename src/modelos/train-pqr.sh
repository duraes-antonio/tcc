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

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"

NUM_ITERATIONS=1000

python3 "${WORK_DIR}/train.py" \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=400,400 \
  --min_resize_value=200 \
  --max_resize_value=500 \
  --resize_factor=16 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=True \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --dataset="pqr" \
  --optimizer="adam" \
  --base_learning_rate=0.01 \
  --adam_learning_rate=0.01 \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=False
