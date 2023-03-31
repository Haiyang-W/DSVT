bash scripts/dist_train.sh 8 --cfg_file ./cfgs/dsvt_models/dsvt_plain_12eD5.yaml --ckpt_save_interval 12 --workers 1 \
--sync_bn --batch_size 8 --epochs 12 --logger_iter_interval 1000