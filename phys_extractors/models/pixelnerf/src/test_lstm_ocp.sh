model_save_name="pixelnerf_lstm"
model_name="physion_model.PN_LSTM"
gpu="0,1,2,3,4,5,6,7"
save_dir="/ccn2/u/thekej/models/"
mode="ocp"
batch_size=1
model_path="/ccn2/u/thekej/pixelnerf_lstm/checkpoint_final.pt"
data_root_path="/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/"
cmd="physion_feature_extract --model_save_name $model_save_name --model $model_name --gpu $gpu --dir_for_saving $save_dir --mode $mode --data_root_path $data_root_path --model_path $model_path --batch_size $batch_size"
echo $cmd
eval "$cmd"

save_dir_model="$save_dir/$model_save_name/$mode"

cmd="physion_train_readout --data-path $save_dir_model/train_features.hdf5 --data-type $model_save_name'_'$mode  --test-path $save_dir_model/test_features.hdf5 --scenario-name observed --train-scenario-indices $save_dir_model/train_json.json --test-scenario-indices $save_dir_model/test_json.json  --scenario features --test-scenario-map $save_dir_model/test_scenario_map.json --one-scenario all --ocp --save_path $save_dir_model"
echo $cmd
eval "$cmd"




model_save_name="pixelnerf_lstm"
model_name="physion_model.PN_LSTM_SIM"
gpu="0,1,2,3,4,5,6,7"
save_dir="/ccn2/u/thekej/models/"
mode="sim"
batch_size=1
model_path="/ccn2/u/thekej/pixelnerf_lstm/checkpoint_final.pt"
data_root_path="/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/"
cmd="physion_feature_extract --model_save_name $model_save_name --model $model_name --gpu $gpu --dir_for_saving $save_dir --mode $mode --data_root_path $data_root_path --model_path $model_path --batch_size $batch_size"
echo $cmd
eval "$cmd"

save_dir_model="$save_dir/$model_save_name/$mode"

cmd="physion_train_readout --data-path $save_dir_model/train_features.hdf5 --data-type $model_save_name'_'$mode  --test-path $save_dir_model/test_features.hdf5 --scenario-name observed --train-scenario-indices $save_dir_model/train_json.json --test-scenario-indices $save_dir_model/test_json.json  --scenario features --test-scenario-map $save_dir_model/test_scenario_map.json --one-scenario all --ocp --save_path $save_dir_model"
echo $cmd
eval "$cmd"
