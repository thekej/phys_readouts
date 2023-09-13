all_scenarios="collide dominoes drop link roll support contain"

for scenario in $all_scenarios; do

  cmd="python LR_scikit.py --model-type logistic --data-path /ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/train_features.hdf5 --data-type bbnet --test-path /ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/test_features.hdf5 --scenario-name observed --train-scenario-indices /ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/train_json.json --test-scenario-indices /ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/test_json.json --scenario features --test-scenario-map /ccn2/u/thekej/R3M_readout/test_map.json --one-scenario all --all-but-one "

  cmd="$cmd $scenario"

  cmd_tmux="tmux kill-session -t $scenario"
  echo $cmd_tmux
  eval " $cmd_tmux"

  cmd_tmux="tmux new -d -s $scenario"
  echo $cmd_tmux
  eval " $cmd_tmux"

  cmd_conda="tmux send-keys -t $scenario \"conda activate /ccn2/u/rmvenkat/across_conda/envs/vmae_flash_2\" C-m"
  echo $cmd_conda
  eval " $cmd_conda"

  cmd_tmux="tmux send-keys -t $scenario \"$cmd \" C-m"
  echo $cmd_tmux
  eval " $cmd_tmux"

done

