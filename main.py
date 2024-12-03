import utils
from log import initialize_log_file, log
import json
import openai
import string

with open("config.json") as config_file:
    api_key = json.load(config_file)["api_key"]
openai.api_key = api_key


def main():
    args = utils.parse_arguments()
    initialize_log_file(args.dataset, args.method)

    log("*****************************")
    log(args)
    log("*****************************")

    utils.fix_seed(args.random_seed)

    log("setup data loader ...")
    dataloader = utils.setup_data_loader(args)
    log(utils.print_now(return_flag=True))

    correct_list = []
    candidates_all_wrong = []
    candidates_all_correct = []
    for idx, data in enumerate(dataloader):
        if idx >= args.test_dataset_size:
            break

        x, y = data
        question = x[0]
        y = y[0].strip()

        log(
            "***************************************************************************"
        )
        log("{}st data".format(idx + 1))

        if args.dataset in ["aqua", "commonsensqa"]:
            y = "(" + y + ")"
        elif args.dataset in ["gsm8k", "addsub", "multiarith", "svamp", "singleeq"]:
            # 1,000,000 => 1000000
            y = float(y.replace(",", ""))
        elif args.dataset in ["strategyqa"]:
            pass
        else:
            raise ValueError("dataset is not properly defined ...")

        # Prepare question template ...
        x = f"Answer the following math word problem. Problem: {question}. Your final answer should be in the form \\boxed{{answer}}, at the end of your response.\n\n"

        answers, all_correct, all_wrong = generate_candidates(args, x, y)

        if all_correct:
            candidates_all_correct.append(True)
            log("===================All candidates are correct===================")
            for idx, (parsed_answer, _) in enumerate(answers):
                log(f"pred {string.ascii_uppercase[idx]}: {parsed_answer}")
            log(f"GT : {y}")
            continue
        elif all_wrong:
            candidates_all_wrong.append(True)
            log("===================All candidates are wrong!===================")
            for idx, (parsed_answer, _) in enumerate(answers):
                log(f"pred {string.ascii_uppercase[idx]}: {parsed_answer}")
            log(f"GT : {y}")
            continue
        else:
            log("===================Some candidates are correct!===================")

        x = "Read the following math word problem and then read two possible answers A and B, along with their explanations. Only one of the answers is correct. Choose which answer is correct and explain step by step. Your final answer should be in the form \\answer{}, at the end of your response. e.g., \\answer{A} or \\answer{B}."
        x += f"Here is the math problem: {question}.\n\n"
        for i, (answer, _) in enumerate(answers):
            x += f"Here is candidate answer {string.ascii_uppercase[i]}: {answer}\n\n"
        x += "Explain step by step and pick a single correct answer. Make sure the final answer has the proper format, \\answer{}."

        history = [x]

        # Answer prediction by generating text ...
        z = utils.generate(args.model, args.api_time_interval, history, args.max_length, temp=0)

        pred = z
        log(x + "\n===================================================\n" + pred)
        try:
            answer_num = utils.parse_compared_answer(pred)
            correct = answers[answer_num][1]
        except:
            log("Parse failed.")
            correct = False

        log(f"({'Correct' if correct else 'Wrong'})")
        log(f"GT : {y}")
        log(
            "***************************************************************************"
        )

        correct_list.append(correct)

    # Calculate accuracy ...
    total = args.test_dataset_size
    if correct_list:
        accuracy = (sum(correct_list) * 1.0 / len(correct_list)) * 100
    else:
        accuracy = 0
    all_correct_rate = (sum(candidates_all_correct) / total) * 100
    all_wrong_rate = (sum(candidates_all_wrong) / total) * 100

    log("------------------------------------------------------------------")
    log(f"Overall accuracy : {accuracy} = {sum(correct_list)}/ {len(correct_list)}")
    log(
        f"All candidates are correct: {all_correct_rate} = {sum(candidates_all_correct)} / {total}"
    )
    log(
        f"All candidates are wrong: {all_wrong_rate} = {sum(candidates_all_wrong)} / {total}"
    )


def generate_candidates(args, x, y):
    answers = []
    for _ in range(args.num_candidates):
        answer = utils.generate(
            args.model, args.api_time_interval, [x], args.max_length, temp=1.2
        )

        try:
            parsed_answer = utils.parse_answer(answer)
            # cleaned_answer = utils.answer_cleansing(args, parsed_answer)
            is_answer_correct = utils.is_correct(parsed_answer, y)

            answer = utils.replace_answer(answer)
        except:
            log("Parse failed.")
            is_answer_correct = False

        answers.append((answer, is_answer_correct))

    all_correct = all([is_correct for _, is_correct in answers])
    all_wrong = not any([is_correct for _, is_correct in answers])
    return answers, all_correct, all_wrong


if __name__ == "__main__":
    main()
