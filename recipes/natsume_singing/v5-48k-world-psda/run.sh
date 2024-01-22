#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/../../../
NNSVS_COMMON_ROOT=$NNSVS_ROOT/recipes/_common/spsvs
NO2_ROOT=$NNSVS_ROOT/recipes/_common/no2
. $NNSVS_ROOT/utils/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($dev_set $eval_set)

dumpdir=dump
dump_org_dir=$dumpdir/$spk/org
dump_norm_dir=$dumpdir/$spk/norm

stage=0
stop_stage=0

. $NNSVS_ROOT/utils/parse_options.sh || exit 1;

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}
else
    expname=${spk}_${tag}
fi
expdir=exp/$expname

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e $db_root ]; then
	cat<<EOF
stage -1: Downloading

This recipe does not download Natsume_Singing_DB.zip to
provide you the opportunity to read the original license.

Please visit https://ksdcm1ng.wixsite.com/njksofficial and read the term of services,
and then download the singing voice database manually.
EOF
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    sh $NO2_ROOT/utils/data_prep.sh ./config.yaml musicxml
    mkdir -p data/list

    echo "train/dev/eval split"
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
	| grep -v _low | sort > data/list/utt_list.txt
    grep 50 data/list/utt_list.txt > data/list/$eval_set.list
    grep 51 data/list/utt_list.txt > data/list/$dev_set.list
    grep -v -e 50 -e 51 -e 23_seg3 data/list/utt_list.txt > data/list/$train_set.list
fi

# Run the rest of the steps
# Please check the script file for more details
. $NNSVS_COMMON_ROOT/run_common_steps_dev.sh

if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
    echo "stage 101: Feature generation"
    . $NNSVS_COMMON_ROOT/feature_generation2.sh
fi

if [ ${stage} -le 104 ] && [ ${stop_stage} -ge 104 ]; then
    echo "stage 104: Training acoustic model"
    . $NNSVS_COMMON_ROOT/train_acoustic2.sh
fi

if [ ${stage} -le 999 ] && [ ${stop_stage} -ge 999 ]; then
    echo "Pack models for SVS(Modified version)"
    # PWG
    if [[ -z "${vocoder_eval_checkpoint}" && -f ${expdir}/${vocoder_model}/config.yml ]]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    # uSFGAN
    elif [[ -z "${vocoder_eval_checkpoint}" && -f ${expdir}/${vocoder_model}/config.yaml ]]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    fi
    # Determine the directory name of a packed model
    if [ -e "$vocoder_eval_checkpoint" ]; then
        # PWG's expdir or packed model's dir
        voc_dir=$(dirname $vocoder_eval_checkpoint)
        # PWG's expdir
        if [ -e ${voc_dir}/config.yml ]; then
            voc_config=${voc_dir}/config.yml
            vocoder_config_name=$(basename $(grep config: ${voc_config} | awk '{print $2}'))
            vocoder_config_name=${vocoder_config_name/.yaml/}
        # uSFGAN
        elif [ -e ${voc_dir}/config.yaml ]; then
            voc_config=${voc_dir}/config.yaml
            vocoder_config_name=$(basename $(grep out_dir: ${voc_config} | awk '{print $2}'))
        # Packed model's dir
        elif [ -e ${voc_dir}/vocoder_model.yaml ]; then
            # NOTE: assuming PWG for now
            voc_config=${voc_dir}/vocoder_model.yaml
            vocoder_config_name=$(basename $(grep config: ${voc_config} | awk '{print $2}'))
            vocoder_config_name=${vocoder_config_name/.yaml/}
        else
            echo "ERROR: vocoder config is not found!"
            exit 1
        fi
        dst_dir=packed_models/${expname}_${timelag_model}_${duration_model}_${acoustic_model}_${vocoder_config_name}
    else
        dst_dir=packed_models/${expname}_${timelag_model}_${duration_model}_${acoustic_model}
    fi

    if [[ ${acoustic_features} == *"melf0"* ]]; then
        feature_type="melf0"
    else
        feature_type="world"
    fi

    mkdir -p $dst_dir
    # global config file
    # NOTE: New residual F0 prediction models require relative_f0 to be false.
    cat > ${dst_dir}/config.yaml <<EOL
# Global configs
sample_rate: ${sample_rate}
frame_period: 5
log_f0_conditioning: true
use_world_codec: true
feature_type: ${feature_type}

# Model-specific synthesis configs
timelag:
    allowed_range: ${timelag_allowed_range}
    allowed_range_rest: ${timelag_allowed_range_rest}
    force_clip_input_features: true
duration:
    force_clip_input_features: true
acoustic:
    subphone_features: "coarse_coding"
    force_clip_input_features: true
    relative_f0: false
    post_filter: true
EOL

    . $NNSVS_COMMON_ROOT/pack_model.sh
fi
