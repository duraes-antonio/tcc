CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="${WORK_DIR}/datasets"

PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
TRAIN_LOGDIR="${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"

python3 deeplab/export_model.py \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=8 \
  --num_classes=5 \
  --decoder_output_stride=8 \
  --crop_size=600 \
  --crop_size=600 \
  --dataset="PQR" \
  --checkpoint_path="${TRAIN_LOGDIR}/model.ckpt-1162" \
  --export_path="${WORK_DIR}/frozen_inference_graph.pb"