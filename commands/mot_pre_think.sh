export OPENAI_API_BASE=https://api.chatanywhere.com.cn/v1
#for dataset in drop boolq fact_checker qa_wikidata com_e com_v anli_a1 hotpot_qa




export CUDA_VISIBLE_DEVICES=1,2,3
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
demo_seed=$RANDOM
temperature=1.2
lm_model=gpt-3.5-turbo-0301
query_encoding=x
demo_encoding=x
do_not_retrieve_same_premise_demos=0
lm_format_requirement_at_last=1
shuffle_demos_for_lm_retrieval=0
demos_for_retrieval_using_purely_question=1
how_to_divide_demos_for_retrieval=score_mod
do_not_retrieve_same_premise_demo_with_test=1
num_demo=4
turbo_system_message="You are a helpful assistant. You need to summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
#turbo_system_message="none"
retrieval_hybrid_with_task_demos="none"




method=few_shot_cot
clustered_retrieval=0
decoding_method=self_consistency

demo_pool_from=gt
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
retriever_name="all-mpnet-base-v2"



output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_mot.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_mot.py \
--how_to_divide_demos_for_retrieval $how_to_divide_demos_for_retrieval \
--lm_format_requirement_at_last $lm_format_requirement_at_last \
--shuffle_demos_for_lm_retrieval $shuffle_demos_for_lm_retrieval \
--demos_for_retrieval_using_purely_question $demos_for_retrieval_using_purely_question \
--turbo_system_message "$turbo_system_message" \
--retrieval_hybrid_with_task_demos $retrieval_hybrid_with_task_demos \
--do_not_retrieve_same_premise_demo_with_test $do_not_retrieve_same_premise_demo_with_test \
--do_not_retrieve_same_premise_demos $do_not_retrieve_same_premise_demos \
--exp_tag lm_retrieval_grid \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 100 \
--api_time_interval 0.01 \
--dataset $dataset \
--demo_path manual_demos_transformed/${dataset}.jsonl \
--output_dir $output_dir \
--model $lm_model \
--limit_dataset_size 0 \
--method $method \
--query_encoding $query_encoding \
--demo_encoding $demo_encoding \
--retriever_name $retriever_name \
--demo_sampling_seed $demo_seed \
--clustered_retrieval $clustered_retrieval \
--num_demo $num_demo \
--exp_name $exp_name \
--inference_split train \
--temperature $temperature \
--top_p 1 \
--decoding_method $decoding_method \
--self_consistency_paths $self_consistency_path \
|tee $log_fp



python data_process/extract_demos_from_inference_result.py \
--filtering_criteria entropy \
--inp_fp $output_dir \
--entropy_threshold $entropy_threshold \
--filter_no_trigger 1