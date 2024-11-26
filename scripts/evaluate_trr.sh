split=mini_val # test OR mini_val
model=gpt4 # gpt3.5 or gpt4 
start=1
end=5
num_gen_samples=1

exp_name=${model}_${split}
output_path=outputs/${exp_name}_4trr

python src/evaluate.py --save_gen_path $output_path --eval_split $split --save_results_path ./code_outputs.txt > std_out.txt