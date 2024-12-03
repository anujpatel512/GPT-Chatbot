mkdir -p rci_log

python main.py --method=zero_shot --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot.log
python main.py --method=zero_shot --rci --model=chatgpt --dataset=addsub > rci_log/addsub_rci.log
python main.py --method=zero_shot_cot --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot_cot.log
python main.py --method=zero_shot_cot --rci --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot_cot_rci.log
python main.py --method=few_shot_cot --model=chatgpt --dataset=addsub > rci_log/addsub_few_shot_cot.log
python main.py --method=few_shot_cot --rci --model=chatgpt --dataset=addsub > rci_log/addsub_few_shot_cot_rci.log