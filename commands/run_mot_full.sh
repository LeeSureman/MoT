entropy_threshold=0.3
pre_thinking_self_consistency_path=16

export dataset=$dataset
export entropy_threshold=$entropy_threshold
export self_consistency_path=$pre_thinking_self_consistency_path

bash commands/mot_pre_think.sh
bash commands/mot_test_recall.sh