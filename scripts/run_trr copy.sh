split=mini_val # test OR mini_val
model=gpt4 # gpt3.5 or gpt4 
start=0
end=10
num_gen_samples=1

prompt=prompts/trr.txt 
exp_name=${model}_${split}
output_path=outputs/${exp_name}_trr 

python src/generate.py --output_path $output_path --prompt_file $prompt --split $split --model $model --start $start --end $end --num_gen_samples $num_gen_samples 