split=mini_val # test OR mini_val
model=gpt4 # gpt3.5 or gpt4 
start=0
end=1
num_gen_samples=1

prompt1=prompts/trr1.txt
prompt2=prompts/trr2.txt
prompt3=prompts/trr3.txt
exp_name=${model}_${split}
output_path=outputs/${exp_name}_4trr 

python src/generate.py --output_path $output_path --prompt_file $prompt1 $prompt2 $prompt3 --split $split --model $model --start $start --end $end --num_gen_samples $num_gen_samples > std_out.txt