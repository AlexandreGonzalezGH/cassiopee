lower=8600
upper=8800
port=$((RANDOM % (upper - lower + 1) + lower))

gpu=0
configname="configs/mambatalk.yaml"
modelname="mambatalk"
desc="en_2"
filename="$(date +%m%d_%H%M%S)_${modelname}_${desc}"

CUDA_VISIBLE_DEVICES=$gpu \
python train.py \
--config $configname \
--model $modelname \
--test_start_epoch 25 \
--test_period 25 \
--port $port 