if [ $# -gt 0 ]; then
  echo "Wating for process $1 finish ..."
  tail --pid=$1 -f /dev/null
  sleep 60s
fi

bash scripts/dist_train.sh 8 --cfg_file ./cfgs/dsvt_models/dsvt_3D_D512e.yaml --ckpt_save_interval 12 --workers 1  \
--sync_bn --batch_size 8 --epochs 12 --logger_iter_interval 1000 --extra_tag D512eb8