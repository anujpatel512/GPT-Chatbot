import os

log_file = None  # Global variable for the log file


def initialize_log_file(dataset, method):
    global log_file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, f"{dataset}_{method}.txt")
    log_file = open(log_file_path, "w")


def log(message):
    global log_file
    print(message, flush=True)  # Optional: print the message to the console
    log_file.write(str(message) + "\n")
    log_file.flush()  # Flush the buffer to ensure immediate write
