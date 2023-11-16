entropy_threshold=0.3

export dataset=$dataset
export entropy_threshold=$entropy_threshold
export memory_filter_input_fp=mot_generated_thoughts/${dataset}.jsonl


echo "start filtering memory"
bash commands/mot_filter_memory.sh
echo "finish filtering memory"
bash commands/mot_test_recall.sh