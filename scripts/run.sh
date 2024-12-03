mkdir -p rci_log

python main.py --method=zero_shot --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_rci.log
python main.py --method=zero_shot_cot --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_zero_shot_cot.log
python main.py --method=zero_shot_cot --rci --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_zero_shot_cot_rci.log
python main.py --method=few_shot_cot --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_few_shot_cot.log
python main.py --method=few_shot_cot --rci --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_few_shot_cot_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=multiarith > rci_log/multiarith_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=multiarith > rci_log/multiarith_rci.log
python main.py --method=zero_shot_cot --model=chatgpt --dataset=multiarith > rci_log/multiarith_zero_shot_cot.log
python main.py --method=zero_shot_cot --rci --model=chatgpt --dataset=multiarith > rci_log/multiarith_zero_shot_cot_rci.log
python main.py --method=few_shot_cot --model=chatgpt --dataset=multiarith > rci_log/multiarith_few_shot_cot.log
python main.py --method=few_shot_cot --rci --model=chatgpt --dataset=multiarith > rci_log/multiarith_few_shot_cot_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=aqua > rci_log/aqua_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=aqua > rci_log/aqua_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=commonsensqa > rci_log/commonsensqa_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=commonsensqa > rci_log/commonsensqa_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=addsub > rci_log/addsub_rci.log
python main.py --method=zero_shot_cot --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot_cot.log
python main.py --method=zero_shot_cot --rci --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot_cot_rci.log
python main.py --method=few_shot_cot --model=chatgpt --dataset=addsub > rci_log/addsub_few_shot_cot.log
python main.py --method=few_shot_cot --rci --model=chatgpt --dataset=addsub > rci_log/addsub_few_shot_cot_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=strategyqa > rci_log/strategyqa_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=strategyqa > rci_log/strategyqa_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=svamp > rci_log/svamp_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=svamp > rci_log/svamp_rci.log
python main.py --method=zero_shot_cot --model=chatgpt --dataset=svamp > rci_log/svamp_zero_shot_cot.log
python main.py --method=zero_shot_cot --rci --model=chatgpt --dataset=svamp > rci_log/svamp_zero_shot_cot_rci.log
python main.py --method=few_shot_cot --model=chatgpt --dataset=svamp > rci_log/svamp_few_shot_cot.log
python main.py --method=few_shot_cot --rci --model=chatgpt --dataset=svamp > rci_log/svamp_few_shot_cot_rci.log

python main.py --method=zero_shot --model=chatgpt --dataset=singleeq > rci_log/singleeq_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=singleeq > rci_log/singleeq_rci.log
python main.py --method=zero_shot_cot --model=chatgpt --dataset=singleeq > rci_log/singleeq_zero_shot_cot.log
python main.py --method=zero_shot_cot --rci --model=chatgpt --dataset=singleeq > rci_log/singleeq_zero_shot_cot_rci.log
python main.py --method=few_shot_cot --model=chatgpt --dataset=singleeq > rci_log/singleeq_few_shot_cot.log
python main.py --method=few_shot_cot --rci --model=chatgpt --dataset=singleeq > rci_log/singleeq_few_shot_cot_rci.log

