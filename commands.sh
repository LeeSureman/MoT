for dataset in gsm8k aqua strategyqa_small
do

# dataset=strategy_small
num_demo=4
clustered_retrieval=4

demo_seed=1

now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
query_encoding=x
demo_encoding=x
method=my_random_sample_few_shot_cot
#method=retrieval_few_shot_cot
lm_model=gpt-3.5-turbo-0301
retriever_name=hkunlp/instructor-large
output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_retrieval.py \
--dataset $dataset \
--demo_path demos/$dataset/$exp_name \
--output_dir $output_dir \
--model $lm_model \
--limit_dataset_size 600 \
--method $method \
--query_encoding $query_encoding \
--demo_encoding $demo_encoding \
--clustered_retrieval 0 \
--retriever_name $retriever_name \
--demo_sampling_seed $demo_seed \
--clustered_retrieval $clustered_retrieval \
--num_demo $num_demo \
--exp_name $exp_name \
--inference_split test \
--temperature 1 \
--top_p 1 \
--decoding_method greedy \
--self_consistency_paths 16 \
|tee $log_fp

done

123

for dataset in aqua openbookqa commonsensqa nli gsm8k strategyqa
do
  python tmp_data_process/create_manual_demo_json.py --dataset $dataset
done


demo_seed=0
dataset=anli_a2
decoding_method=greedy
num_demo=4
clustered_retrieval=0
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
query_encoding=x
demo_encoding=x
method=few_shot_cot
lm_model=gpt-3.5-turbo-0301
retriever_name=hkunlp/instructor-large
output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_retrieval.py \
--api_time_interval 0.01 \
--dataset $dataset \
--demo_path manual_demos_transformed/nli.jsonl \
--output_dir $output_dir \
--model $lm_model \
--limit_dataset_size 600 \
--method $method \
--query_encoding $query_encoding \
--demo_encoding $demo_encoding \
--retriever_name $retriever_name \
--demo_sampling_seed $demo_seed \
--clustered_retrieval $clustered_retrieval \
--num_demo $num_demo \
--exp_name $exp_name \
--inference_split test \
--temperature 1 \
--top_p 1 \
--decoding_method $decoding_method \
--self_consistency_paths 16 \
|tee $log_fp

for trial in 1
do
for decoding_method in self_consistency
do
for dataset in anli_a2 anli_a3 gsm8k aqua openbookqa commonsensqa strategyqa_small
do
for lm_model in gpt-3.5-turbo-0301
do
num_demo=4
clustered_retrieval=0
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
query_encoding=x
demo_encoding=x
method=few_shot_cot
#lm_model=text-davinci-003
retriever_name=hkunlp/instructor-large
output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_retrieval.py \
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
--temperature 1 \
--top_p 1 \
--decoding_method $decoding_method \
--self_consistency_paths 32 \
|tee $log_fp
done
done
done
done


decoding_method=greedy
lm_model=gpt-3.5-turbo-0301
dataset=anli_a3
num_demo=4
clustered_retrieval=0
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
query_encoding=x
demo_encoding=x
method=few_shot_cot
#lm_model=text-davinci-003
retriever_name=hkunlp/instructor-large
output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_retrieval.py \
--multi_thread 0 \
--api_time_interval 0.01 \
--dataset $dataset \
--demo_path manual_demos_transformed/${dataset}.jsonl \
--output_dir $output_dir \
--model $lm_model \
--limit_dataset_size 600 \
--method $method \
--query_encoding $query_encoding \
--demo_encoding $demo_encoding \
--retriever_name $retriever_name \
--demo_sampling_seed $demo_seed \
--clustered_retrieval $clustered_retrieval \
--num_demo $num_demo \
--exp_name $exp_name \
--inference_split train \
--temperature 1 \
--top_p 1 \
--decoding_method $decoding_method \
--self_consistency_paths 32 \
|tee $log_fp

12345

#lm_model=text-davinci-003


#在有LM_retrieval之前的命令
for trial in 1
do
for clustered_retrieval in 0
do
for dataset in anli_a2 anli_a3 gsm8k aqua openbookqa commonsensqa strategyqa_small
do
for do_not_retrieve_same_premise_demo_with_test in 1
do
for demo_c in confidence
do
for method in retrieval_few_shot_cot
do
for retrieval_hybrid_with_task_demos in none
do

for retriever_name in hkunlp/instructor-large
do

for turbo_system_message in "You are a helpful assistant." "You are a helpful assistant and you can answer questions based on the examples provided by the user." "You are a helpful assistant and you can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes."
do

temperature=0
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
decoding_method=greedy
lm_model=gpt-3.5-turbo-0301
#lm_model=text-davinci-003
#retriever_name=hkunlp/instructor-large
#retriever_name=all-MiniLM-L6-v2
#retriever_name=all-mpnet-base-v2
#clustered_retrieval=0
demo_pool_from=lm_inference
#demo_pool_from=gt
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
#method=few_shot_cot
query_encoding=x
demo_encoding=x
num_demo=4
demo_seed=$RANDOM
do_not_retrieve_same_premise_demos=0

output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_retrieval.py \
--turbo_system_message "$turbo_system_message" \
--retrieval_hybrid_with_task_demos $retrieval_hybrid_with_task_demos \
--do_not_retrieve_same_premise_demo_with_test $do_not_retrieve_same_premise_demo_with_test \
--do_not_retrieve_same_premise_demos $do_not_retrieve_same_premise_demos \
--exp_tag test_auto_cot_retriever \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 100 \
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
--self_consistency_paths 32 \
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


python generate_relevant_question_prompt.py --lm_inference_result_fp ../V6/experiment/gsm8k/retrieval_2023_03_27_23_09_39s___550___27621/lm_inference_result.jsonl

python generate_relevant_question_prompt.py --lm_inference_result_fp ../V6/experiment/strategyqa_small/retrieval_2023_03_28_04_45_37s___768___21168/lm_inference_result.jsonl

python generate_relevant_question_prompt.py --lm_inference_result_fp ../V6/experiment/commonsensqa/retrieval_2023_03_28_04_38_33s___730___4310/lm_inference_result.jsonl


python generate_question_retrieval_prompt.py \
  --candidate_pair_fp tmp_candidate_pairs_for_retrieval_prompting/commonsensqa_labeled.jsonl \
  --num_retrieval_demos 0 \
  --need_explanation 0 \
  --use_retrieval_demos 1 \
  --just_random_select 0 \
  --seed 1227 \
  --num_candidate_q 15


for lxn in "You are a helpful assistant" "You are a helpful 2"
do
  echo 2
done



#在有LM_retrieval之后的命令
for trial in 1
do
for clustered_retrieval in 4
do
for dataset in anli_a2 anli_a3 gsm8k
do
for do_not_retrieve_same_premise_demo_with_test in 1
do
for demo_c in confidence gt
do
for method in lm_retrieval_few_shot_cot
do
for retrieval_hybrid_with_task_demos in none
do

for retriever_name in all-MiniLM-L6-v2 hkunlp/instructor-large hkunlp/instructor-base
do

#for turbo_system_message in "You are a helpful assistant." "You are a helpful assistant and you can answer questions based on the examples provided by the user." "You are a helpful assistant and you can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes."
#do
for turbo_system_message in "None"
do

for demos_for_retrieval_using_purely_question in 0 1
do

for shuffle_demos_for_lm_retrieval in 0
do

#for how_to_divide_demos_for_retrieval in score_division score_mod
#do


export CUDA_VISIBLE_DEVICES=2,3
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
decoding_method=greedy
temperature=0
lm_model=gpt-3.5-turbo-0301
query_encoding=x
demo_encoding=x
#lm_model=text-davinci-003
#retriever_name=hkunlp/instructor-large
#retriever_name=all-MiniLM-L6-v2
#retriever_name=all-mpnet-base-v2
#clustered_retrieval=0
demo_pool_from=lm_inference
#demo_pool_from=gt
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
#method=few_shot_cot
num_demo=4
demo_seed=$RANDOM
do_not_retrieve_same_premise_demos=0
lm_format_requirement_at_last=1
#shuffle_demos_for_lm_retrieval=0
#demos_for_retrieval_using_purely_question=0
#how_to_divide_demos_for_retrieval=score_division


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
--exp_tag lm_retrieval_grid \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 100 \
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
--self_consistency_paths 32 \
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
done
#done






#把每个数据集的embedding cache好
for trial in 1
do
for clustered_retrieval in 0
do
for dataset in aqua openbookqa strategyqa_small commonsensqa anli_a2 anli_a3
do
for do_not_retrieve_same_premise_demo_with_test in 1
do
for demo_c in gt
do
for method in retrieval_few_shot_cot
do
for retrieval_hybrid_with_task_demos in none
do

for retriever_name in all-MiniLM-L6-v2 hkunlp/instructor-large hkunlp/instructor-base
do

#for turbo_system_message in "You are a helpful assistant." "You are a helpful assistant and you can answer questions based on the examples provided by the user." "You are a helpful assistant and you can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes."
#do
for turbo_system_message in "None"
do

for demos_for_retrieval_using_purely_question in 0
do

for shuffle_demos_for_lm_retrieval in 0
do

for how_to_divide_demos_for_retrieval in score_division
do


export CUDA_VISIBLE_DEVICES=1,2,3
now_time_tag=`date +"%Y_%m_%d_%H_%M_%Ss___%3N"`
exp_name=retrieval_${now_time_tag}___$RANDOM
decoding_method=greedy
temperature=0
lm_model=gpt-3.5-turbo-0301
query_encoding=x
demo_encoding=x
#lm_model=text-davinci-003
#retriever_name=hkunlp/instructor-large
#retriever_name=all-MiniLM-L6-v2
#retriever_name=all-mpnet-base-v2
#clustered_retrieval=0
demo_pool_from=lm_inference
#demo_pool_from=gt
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
#method=few_shot_cot
num_demo=4
demo_seed=$RANDOM
do_not_retrieve_same_premise_demos=0
lm_format_requirement_at_last=1
#shuffle_demos_for_lm_retrieval=0
#demos_for_retrieval_using_purely_question=0
#how_to_divide_demos_for_retrieval=score_division


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
--exp_tag lm_retrieval_grid \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 100 \
--api_time_interval 0.01 \
--dataset $dataset \
--demo_path manual_demos_transformed/${dataset}.jsonl \
--output_dir $output_dir \
--model $lm_model \
--limit_dataset_size 5 \
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
--self_consistency_paths 32 \
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
done
done

python tmp_data_process/create_manual_demo_json.py --dataset drop








#生成新数据集的train的cot

#for dataset in drop boolq fact_checker qa_wikidata com_e com_v anli_a1 hotpot_qa
for dataset in hotpot_qa

do

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
#dataset=hotpot_qa
decoding_method=greedy

demo_pool_from=gt
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
retriever_name="hkunlp/instructor-base"



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
--self_consistency_paths 16 \
|tee $log_fp

done

12345

python tmp_data_process/extract_demos_from_inference_result.py \
--filtering_criteria confidence \
--inp_fp experiment/hotpot_qa/retrieval_2023_04_13_12_22_29s___950___28586/lm_inference_result.jsonl		 \
--confidence_threshold 0.85


for inp_fp in experiment/strategyqa_small/retrieval_2023_03_28_04_45_37s___768___21168/lm_inference_result.jsonl experiment/commonsensqa/retrieval_2023_03_28_04_38_33s___730___4310/lm_inference_result.jsonl experiment/openbookqa/retrieval_2023_03_28_04_33_48s___140___784/lm_inference_result.jsonl experiment/aqua/retrieval_2023_03_27_23_26_52s___721___2205/lm_inference_result.jsonl experiment/gsm8k/retrieval_2023_03_27_23_09_39s___550___27621/lm_inference_result.jsonl experiment/anli_a3/retrieval_2023_03_27_21_33_50s___100___17340/lm_inference_result.jsonl experiment/anli_a2/retrieval_2023_03_27_20_47_21s___099___23810/lm_inference_result.jsonl
do
python tmp_data_process/extract_demos_from_inference_result.py --filtering_criteria confidence --inp_fp $inp_fp --confidence_threshold 0.85
done


#for dataset in aqua drop anli_a1 anli_a2 anli_a3 openbookqa com_v boolq fact_checker qa_wikidata
#for retriever_name in all-MiniLM-L6-v2 hkunlp/instructor-large hkunlp/instructor-base all-mpnet-base-v2


for retriever_name in hkunlp/instructor-base
do
for dataset in aqua drop anli_a1 anli_a2 anli_a3 openbookqa com_v boolq fact_checker qa_wikidata
do

for method in few_shot_cot
do

for clustered_retrieval in 4
do


export CUDA_VISIBLE_DEVICES=4,5,6
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
#turbo_system_message="You are a helpful assistant. You need to summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
#turbo_system_message="none"
#turbo_system_message="You are a helpful assistant."
turbo_system_message="You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes. You must summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
retrieval_hybrid_with_task_demos="none"




#method=few_shot
#clustered_retrieval=4
#dataset=hotpot_qa
decoding_method=greedy

demo_pool_from=lm_inference
demo_c=confidence
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
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
--exp_tag fix_concat_bug \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 100 \
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
--self_consistency_paths 16 \
|tee $log_fp

done
done
done
done



#zero_shot

#for dataset in aqua drop anli_a1 anli_a2 anli_a3 openbookqa com_v boolq fact_checker qa_wikidata
for dataset in boolq fact_checker drop
do

for method in zero_shot zero_shot_cot
do


export CUDA_VISIBLE_DEVICES=4,5,6
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
#turbo_system_message="You are a helpful assistant. You need to summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
turbo_system_message="none"
#turbo_system_message="You are a helpful assistant."
#turbo_system_message="You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes. You must summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
retrieval_hybrid_with_task_demos="none"




#method=zero_shot
clustered_retrieval=0
#dataset=hotpot_qa
decoding_method=greedy

demo_pool_from=lm_inference
demo_c=confidence
demo_pool_path=demos/filter_by_${demo_c}/${dataset}.jsonl
retriever_name="hkunlp/instructor-base"



output_dir=experiment/$dataset/${exp_name}
mkdir $output_dir -p
log_fp=$output_dir/run_inference_retrieval.log
output_dir=$output_dir/lm_inference_result.jsonl

python -u run_inference_zero_shot.py \
--how_to_divide_demos_for_retrieval $how_to_divide_demos_for_retrieval \
--lm_format_requirement_at_last $lm_format_requirement_at_last \
--shuffle_demos_for_lm_retrieval $shuffle_demos_for_lm_retrieval \
--demos_for_retrieval_using_purely_question $demos_for_retrieval_using_purely_question \
--turbo_system_message "$turbo_system_message" \
--retrieval_hybrid_with_task_demos $retrieval_hybrid_with_task_demos \
--do_not_retrieve_same_premise_demo_with_test $do_not_retrieve_same_premise_demo_with_test \
--do_not_retrieve_same_premise_demos $do_not_retrieve_same_premise_demos \
--exp_tag zero_shot_exp \
--demo_pool_from $demo_pool_from \
--demo_pool_path $demo_pool_path \
--multi_thread 100 \
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
--self_consistency_paths 16 \
|tee $log_fp

done
done

python tmp_data_process/extract_demos_from_inference_result_new.py \
--filtering_criteria max_p \
--inp_fp ../V8/experiment/drop/retrieval_2023_04_13_12_17_15s___839___16269/lm_inference_result.jsonl		 \
--max_p_threshold 0.85

python tmp_data_process/extract_demos_from_inference_result_new.py \
--filtering_criteria entropy \
--inp_fp ../V8/experiment/drop/retrieval_2023_04_13_12_17_15s___839___16269/lm_inference_result.jsonl		 \
--max_p_threshold 0.85 \
--entropy_threshold 0.55



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


for inp_fp in ../V8/experiment/drop/retrieval_2023_04_13_12_17_15s___839___16269/lm_inference_result.jsonl ../V8/experiment/boolq/retrieval_2023_04_12_23_23_41s___147___12045/lm_inference_result.jsonl ../V8/experiment/fact_checker/retrieval_2023_04_12_23_30_30s___496___29257/lm_inference_result.jsonl ../V8/experiment/qa_wikidata/retrieval_2023_04_13_11_03_58s___207___22489/lm_inference_result.jsonl ../V8/experiment/com_v/retrieval_2023_04_12_23_52_14s___446___13528/lm_inference_result.jsonl ../V8/experiment/anli_a1/retrieval_2023_04_12_23_59_00s___845___18161/lm_inference_result.jsonl ../V4/experiment/anli_a2/retrieval_2023_03_27_20_47_21s___099___23810/lm_inference_result.jsonl ../V4/experiment/anli_a3/retrieval_2023_03_27_21_33_50s___100___17340/lm_inference_result.jsonl ../V4/experiment/aqua/retrieval_2023_03_27_23_26_52s___721___2205/lm_inference_result.jsonl ../V4/experiment/openbookqa/retrieval_2023_03_28_04_33_48s___140___784/lm_inference_result.jsonl
do
for filter_no_trigger in 0 1
do

#for max_p_threshold in 0.1 0.15 0.2 0.3 0.35 0.4 0.45 0.5 0.6 0.65 0.7 0.8 0.85 1
#do
#python tmp_data_process/extract_demos_from_inference_result_new.py --filtering_criteria max_p --inp_fp $inp_fp --max_p_threshold $max_p_threshold --filter_no_trigger $filter_no_trigger
#done

#for entropy_threshold in 0.1 0.2 0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.8 0.9 1.1 1.4 1.7 2.0 2.3 2.8
for entropy_threshold in 0.02 0.05 0.08
do
python tmp_data_process/extract_demos_from_inference_result_new.py --filtering_criteria entropy --inp_fp $inp_fp --entropy_threshold $entropy_threshold --filter_no_trigger $filter_no_trigger
done

done
done




exp_tag=new_self_consistncy

for demo_seed in 1
do
for entropy_threshold in 0.3
do
#for dataset in aqua drop anli_a1 anli_a2 anli_a3 openbookqa com_v boolq fact_checker qa_wikidata
for dataset in aqua drop anli_a1 anli_a2 anli_a3 openbookqa com_v boolq fact_checker qa_wikidata
#for dataset in drop anli_a1 anli_a2 com_v boolq
do

#for retriever_name in all-MiniLM-L6-v2 hkunlp/instructor-base all-mpnet-base-v2
for retriever_name in all-mpnet-base-v2
do

for self_consistency_paths in 8 32
do
for temperature in 0.7 1 1.2
do
for method in lm_retrieval_few_shot_cot few_shot_cot
do

for clustered_retrieval in 4
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
#temperature=1.2
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
#turbo_system_message="You are a helpful assistant. You need to summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
#turbo_system_message="none"
#turbo_system_message="You are a helpful assistant."
#turbo_system_message="You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes. You must summarize your answer in the end of your response, with the format \"The answer is ...\", like the provided demonstrations."
#turbo_system_message="You are a helpful assistant. You can answer questions based on the examples provided by the user. If you think there are mistakes in those examples, you can ignore those mistakes."
retrieval_hybrid_with_task_demos="none"




#method=zero_shot
#clustered_retrieval=0
#dataset=hotpot_qa
decoding_method=self_consistency

demo_pool_from=lm_inference
demo_c=entropy
#entropy_threshold=0.5
#max_p=0.85
demo_pool_path=demos/filter_by_${demo_c}/${dataset}_${entropy_threshold}_${filter_no_trigger}.jsonl
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
--multi_thread 100 \
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


#for inp_fp in ../V8/experiment/drop/retrieval_2023_04_13_12_17_15s___839___16269/lm_inference_result.jsonl ../V8/experiment/boolq/retrieval_2023_04_12_23_23_41s___147___12045/lm_inference_result.jsonl ../V8/experiment/fact_checker/retrieval_2023_04_12_23_30_30s___496___29257/lm_inference_result.jsonl ../V8/experiment/qa_wikidata/retrieval_2023_04_13_11_03_58s___207___22489/lm_inference_result.jsonl ../V8/experiment/com_v/retrieval_2023_04_12_23_52_14s___446___13528/lm_inference_result.jsonl ../V8/experiment/anli_a1/retrieval_2023_04_12_23_59_00s___845___18161/lm_inference_result.jsonl ../V4/experiment/anli_a2/retrieval_2023_03_27_20_47_21s___099___23810/lm_inference_result.jsonl ../V4/experiment/anli_a3/retrieval_2023_03_27_21_33_50s___100___17340/lm_inference_result.jsonl ../V4/experiment/aqua/retrieval_2023_03_27_23_26_52s___721___2205/lm_inference_result.jsonl ../V4/experiment/openbookqa/retrieval_2023_03_28_04_33_48s___140___784/lm_inference_result.jsonl
for inp_fp in ../V8/experiment/drop/retrieval_2023_04_13_12_17_15s___839___16269/lm_inference_result.jsonl
do
for filter_no_trigger in 1
do


entropy_threshold=1
#for entropy_threshold in 0.1 0.2 0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.8 0.9 1.1 1.4 1.7 2.0 2.3 2.8

python tmp_data_process/extract_demos_from_inference_result_new.py \
--filtering_criteria gt \
--inp_fp $inp_fp \
--entropy_threshold $entropy_threshold \
--filter_no_trigger $filter_no_trigger \


done
done