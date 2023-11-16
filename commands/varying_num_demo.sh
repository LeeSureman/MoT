exp_tag=varying_retrieved_demo_num
for demo_seed in 5
do
for entropy_threshold in 0.3
do
#for dataset in drop aqua anli_a1 anli_a2 anli_a3 boolq qa_wikidata fact_checker openbookqa com_v
for dataset in drop anli_a1 anli_a2 anli_a3 boolq qa_wikidata fact_checker aqua openbookqa com_v
do
for self_consistency_paths in 16
do
for temperature in 1
do
#for dataset in aqua drop anli_a1 anli_a2 anli_a3 com_v boolq fact_checker qa_wikidata
#for dataset in drop anli_a1 anli_a2 com_v boolq


#for retriever_name in all-MiniLM-L6-v2 hkunlp/instructor-base all-mpnet-base-v2
for retriever_name in all-mpnet-base-v2
do

for method in few_shot_cot lm_retrieval_few_shot_cot
do

for num_demo in 1 2 3
do

for clustered_retrieval in $num_demo
do


#for turbo_system_message in "none" "You are a helpful assistant." "You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes."
for turbo_system_message in "You are a helpful assistant."
do

if [ "$dataset" = "aqua" ] || [ "$dataset" = "fact_checker" ] || [ "$dataset" = "qa_wikidata" ]; then
  filter_no_trigger=0

else
  filter_no_trigger=1

fi

export CUDA_VISIBLE_DEVICES=1,2
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
#demo_seed=$RANDOM
#temperature=1
lm_model=gpt-3.5-turbo-0301
query_encoding=x
demo_encoding=x
do_not_retrieve_same_premise_demos=0
lm_format_requirement_at_last=1
shuffle_demos_for_lm_retrieval=0
demos_for_retrieval_using_purely_question=1
how_to_divide_demos_for_retrieval=score_mod
do_not_retrieve_same_premise_demo_with_test=1
#num_demo=4
#turbo_system_message="You are a helpful assistant. You need to summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
#turbo_system_message="none"
#turbo_system_message="You are a helpful assistant."
#turbo_system_message="You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes. You must summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
#turbo_system_message="You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes."
retrieval_hybrid_with_task_demos="none"




#method=zero_shot
#clustered_retrieval=0
#dataset=hotpot_qa
decoding_method=greedy

demo_pool_from=lm_inference
demo_c=entropy
#entropy_threshold=0.5
#max_p=0.85
demo_pool_path=demos_tmp/filter_by_${demo_c}/${dataset}_${entropy_threshold}_${filter_no_trigger}.jsonl
#retriever_name="hkunlp/instructor-base"



output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_retrieval.py \
--how_to_divide_demos_for_retrieval $how_to_divide_demos_for_retrieval \
--lm_format_requirement_at_last $lm_format_requirement_at_last \
--shuffle_demos_for_lm_retrieval $shuffle_demos_for_lm_retrieval \
--demos_for_retrieval_using_purely_question $demos_for_retrieval_using_purely_question \
--turbo_system_message "$turbo_system_message" \
--retrieval_hybrid_with_task_demos $retrieval_hybrid_with_task_demos \
--do_not_retrieve_same_premise_demo_with_test $do_not_retrieve_same_premise_demo_with_test \
--do_not_retrieve_same_premise_demos $do_not_retrieve_same_premise_demos \
--exp_tag $exp_tag \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 30 \
--api_time_interval 0.01 \
--dataset $dataset \
--demo_path manual_demos_transformed/${dataset}.jsonl \
--output_dir $output_dir \
--model $lm_model \
--limit_dataset_size 1000 \
--method $method \
--query_encoding $query_encoding \
--demo_encoding $demo_encoding \
--retriever_name $retriever_name \
--demo_sampling_seed $demo_seed \
--clustered_retrieval $clustered_retrieval \
--num_demo $num_demo \
--exp_name $exp_name \
--inference_split test \
--temperature $temperature \
--top_p 1 \
--decoding_method $decoding_method \
--self_consistency_paths $self_consistency_paths \
--demo_c $demo_c \
--entropy_threshold $entropy_threshold \
--filter_no_trigger $filter_no_trigger \
|tee $log_fp

done
done
done
done
done
done
done
done
done
done