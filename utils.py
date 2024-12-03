from statistics import mean
import openai
import multiprocessing
import json
import random
import re
import random
import time
import datetime
from torch.utils.data import Dataset
import torch
import numpy as np
from log import log
import argparse

torch.multiprocessing.set_start_method("fork")


def is_correct(y, pred):
    if not y or not pred:
        return False

    try:
        correct = (
            False if not pred else (np.array([pred]) == np.array([y])).sum().item()
        )
    except:
        import code

        code.interact(local=locals())

    return correct


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="aqua",
        choices=[
            "aqua",
            "gsm8k",
            "commonsensqa",
            "addsub",
            "multiarith",
            "strategyqa",
            "svamp",
            "singleeq",
        ],
        help="dataset used for experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chatgpt",
        choices=["chatgpt", "gpt4"],
        help="model used for decoding. Note that 'gpt3' are the smallest models.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="compare",
        choices=["compare"],
        help="method",
    )
    parser.add_argument("--rci", default=False, action="store_true")
    parser.add_argument("--nofeedback", default=False, action="store_true")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="maximum length of output tokens by model for answer extraction",
    )
    parser.add_argument(
        "--test-dataset-size",
        type=int,
        default=10,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.",
    )
    parser.add_argument("--api-time-interval", type=float, default=0.25, help="")
    parser.add_argument("--num-candidates", type=int, default=2)

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = (
            "Choose the most plausible answer from among choices A through E."
        )
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    else:
        raise ValueError("dataset is not properly defined ...")

    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


def parse_compared_answer(input_str):
    pattern = r"\\answer\{(.*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        match_str = re.sub(r"[^A-Z]", "", match_str)
        if match_str[0]:
            # Map A to 0, B to 1, etc.
            solution = ord(match_str) - ord('A')
            break

    return solution if solution is not None else None


def replace_answer(input_str: str):
    pattern = r"(\\boxed\{.*\})"
    matches = re.findall(pattern, input_str)

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)

        if solution:
            input_str = input_str.replace(match_str, solution)

    return input_str


def parse_answer(input_str):
    pattern = r"\\boxed\{(.*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return float(solution) if solution else None


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    now = now.strftime("%Y/%m/%d %H:%M:%S")
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def generate(model: str, api_time_interval: float, history, max_length=None, temp:float=0):
    time.sleep(api_time_interval)

    if model == "chatgpt" or model == "gpt4":
        system_message = "You are a helpful assistant that can help users to answer reasoning questions"
        messages = []
        messages.append({"role": "system", "content": system_message})

        is_user = True
        for text in history:
            if is_user:
                messages.append({"role": "user", "content": text})
            else:
                messages.append({"role": "assistant", "content": text})
            is_user = not is_user

        if is_user:
            raise ValueError("Conversation ends with an assistant turn")

        if model == "chatgpt":
            model = "gpt-3.5-turbo-0613"
        elif model == "gpt4":
            model = "gpt-4"

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_length if max_length else 512,
                )
            except:
                pass
            else:
                break

        return response["choices"][0]["message"]["content"]
    else:
        raise ValueError("model is not properly defined ...")


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    log("dataset : {}".format(args.dataset))
    log("data size : {}".format(len(answers)))
    log("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    log("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, 3)
    log("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        drop_last=False,
        num_workers=dataloader_num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    return dataloader


def answer_cleansing(args, pred):
    if args.method in ["few_shot_cot"]:
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ["aqua", "commonsensqa"]:
        pred = re.findall(r"\(A\)|\(B\)|\(C\)|\(D\)|\(E\)", pred)
    elif args.dataset in ["gsm8k", "addsub", "multiarith", "svamp", "singleeq"]:
        pred = pred.replace(",", "")
        pred = [float(s) for s in re.findall(r"-?\d+\.?\d*", pred)]
    elif args.dataset in ["strategyqa"]:
        pred = pred.lower()
        pred = re.sub("\"|'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ["few_shot_cot"]:
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ["zero_shot_cot"]:
            # choose the first element in list ...
            pred = pred[-1]
        elif args.method in ["zero_shot", "compare"]:
            pred = pred[-1]
        else:
            raise ValueError("method is not properly defined ...")

    return pred


def create_demo_text(args, cot_flag):
    x, z, y = [], [], []

    # example sentences ...
    if args.dataset in ("multiarith", "gsm8k", "addsub", "singleeq", "svamp"):
        x.append(
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        )
        z.append(
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6."
        )
        y.append("6")

        x.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
        )
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")

        x.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
        )
        z.append(
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39."
        )
        y.append("39")

        x.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
        )
        z.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8."
        )
        y.append("8")

        x.append(
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
        )
        z.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9."
        )
        y.append("9")

        x.append(
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
        )
        z.append(
            "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29."
        )
        y.append("29")

        x.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
        )
        z.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls."
        )
        y.append("33")

        x.append(
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
        )
        z.append(
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8."
        )
        y.append("8")
    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += (
                "Q: "
                + x[i]
                + "\nA: "
                + z[i]
                + " "
                + args.direct_answer_trigger_for_fewshot
                + " "
                + y[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Q: "
                + x[i]
                + "\nA: "
                + args.direct_answer_trigger_for_fewshot
                + " "
                + y[i]
                + ".\n\n"
            )

    return demo_text


def do_rci(args, y, max_length, history, z, pred, correct):
    MAX_RCI = 2
    turn = 0
    while args.nofeedback or not correct:
        if turn >= MAX_RCI:
            break

        history.append(z)
        if turn == 0:
            history.append(
                "Your answer is wrong. Review your previous answer and find problems with your answer."
            )
        elif turn == 1:
            history.append(
                "I think your answer still have some errors. Review your answer and find problems with your answer."
            )
        else:
            pass

        z = generate(args, history, max_length)
        history.append(z)

        if turn == 0:
            history.append("Based on the problems you found, improve your answer")
        elif turn == 1:
            history.append("Based on the problems you found, improve your answer")
        else:
            pass

        z = generate(args, history, max_length)

        log("\n".join(history[-3:]) + f"\n{z}")
        updated_pre = answer_cleansing(args, z)
        pred = updated_pre if updated_pre else pred

        correct = is_correct(y, pred)

        turn += 1
    return pred, correct
