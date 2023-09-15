run_in_tmux_session() {
    # Parameters
    local session_name="$1"
    local command_to_run="$2"

    # Kill existing tmux session (if exists)
    local cmd_tmux="tmux kill-session -t '$session_name'"
    echo $cmd_tmux
    eval "$cmd_tmux"

    # Create new tmux session
    cmd_tmux="tmux new -d -s '$session_name'"
    echo $cmd_tmux
    eval "$cmd_tmux"

    # Activate the specific conda environment
    local cmd_conda="tmux send-keys -t '$session_name' \"conda activate /ccn2/u/rmvenkat/across_conda/envs/vmae_flash_2\" C-m"
    echo $cmd_conda
    eval "$cmd_conda"

    # Send the provided command to tmux session
    local cmd_save_features="tmux send-keys -t '$session_name' \"$command_to_run\" C-m"
    echo $cmd_save_features
    eval "$cmd_save_features"
}


exp_dir="/ccn2/u/rmvenkat/data/bbnet_ablations/"

#folder_names="OurMasking_1399 MagViT_masking_1399 MaskViT_masking_1399"

#folder_names="OurMasking_1999"

#folder_names="MagViT_masking_1999"
#
#folder_names="MaskViT_masking_1999"

#folder_names="OurMasking_1999 MagViT_masking_1999 MaskViT_masking_1999"

#folder_names="OurMasking_1399 OurMasking_1699 OurMasking_1999 OurMasking_3279 MagViT_masking_1399 MagViT_masking_1699 MagViT_masking_3279 MagViT_masking_1999 MaskViT_masking_1399 MaskViT_masking_1699 MaskViT_masking_1999 MaskViT_masking_3279"
#
#folder_names="OurMasking_1399_more_unroll OurMasking_1699_more_unroll OurMasking_1999_more_unroll OurMasking_3279_more_unroll OurMasking_2599_more_unroll MagViT_masking_1399_more_unroll MagViT_masking_1699_more_unroll MagViT_masking_3279_more_unroll MagViT_masking_2599_more_unroll MagViT_masking_1999_more_unroll MaskViT_masking_1399_more_unroll MaskViT_masking_1699_more_unroll MaskViT_masking_1999_more_unroll MaskViT_masking_3279_more_unroll MaskViT_masking_2599_more_unroll"


folder_names="OurMasking_lr_1279_more_unroll OurMasking_lr_599_more_unroll OurMasking_lr_999_more_unroll MagViT_lr_999_more_unroll MagViT_lr_599_more_unroll MagViT_lr_1279_more_unroll MaskViT_masking_lr_999_more_unroll MaskViT_masking_lr_599_more_unroll MaskViT_masking_lr_1279_more_unroll"

#folder_names="MagViT_masking_3279_ocd "
#
#folder_names="MaskViT_masking_3279_ocd "
#
#folder_names="MaskViT_masking_1699_ocd "

suffix=""

for folder_name in $folder_names; do
  cmd="python LR_scikit.py --model-type logistic --data-path $exp_dir/$folder_name/train_features.hdf5 --data-type $folder_name$suffix --test-path $exp_dir/$folder_name/test_features.hdf5 --scenario-name observed --train-scenario-indices $exp_dir/$folder_name/train_json.json --test-scenario-indices $exp_dir/$folder_name/test_json.json --scenario features --test-scenario-map /ccn2/u/thekej/R3M_readout/test_map.json --one-scenario all"
  echo $cmd
  eval $cmd
done