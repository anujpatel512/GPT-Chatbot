mkdir -p rci_log

python main.py --method=compare --model=gpt4 --dataset=gsm8k --test-dataset-size 500
# python main.py --method=zero_shot --model=chatgpt --dataset=gsm8k > rci_log/gsm8k_zero_shot.log

# python main.py --method=zero_shot --model=chatgpt --dataset=multiarith > rci_log/multiarith_zero_shot.log


# python main.py --method=zero_shot --model=chatgpt --dataset=aqua > rci_log/aqua_zero_shot.log


# python main.py --method=zero_shot --model=chatgpt --dataset=commonsensqa > rci_log/commonsensqa_zero_shot.log


# python main.py --method=zero_shot --model=chatgpt --dataset=addsub > rci_log/addsub_zero_shot.log


# python main.py --method=zero_shot --model=chatgpt --dataset=strategyqa > rci_log/strategyqa_zero_shot.log


# python main.py --method=zero_shot --model=chatgpt --dataset=svamp > rci_log/svamp_zero_shot.log


# python main.py --method=zero_shot --model=chatgpt --dataset=singleeq > rci_log/singleeq_zero_shot.log


