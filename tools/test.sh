if [ $# -gt 0 ]; then
  echo "Wating for process $1 finish ..."
  tail --pid=$1 -f /dev/null
  sleep 60s
fi

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_test.sh 4 --cfg_file ./cfgs/dsvt_models/dsvt_plain_12eD5.yaml \
# --batch_size 4 --workers 1 \
# --ckpt ../output/cfgs/dsvt_models/dsvt_plain_12eD5/default/ckpt/checkpoint_epoch_12.pth &&
# sleep 60s &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_test.sh 4 --cfg_file ./cfgs/dsvt_models/dsvt_plain_12eD5B8.yaml \
# --batch_size 4 --workers 1 \
# --ckpt ../output/cfgs/dsvt_models/dsvt_plain_12eD5B8/default/ckpt/checkpoint_epoch_12.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_test.sh 4 --cfg_file ./cfgs/dsvt_models/dsvt_plain_12eD5.yaml \
--batch_size 4 --workers 1 \
--ckpt ../output/cfgs/dsvt_models/dsvt_plain_12eD5/default/ckpt/checkpoint_epoch_12.pth