import os
import datetime

def log_and_print(message):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = "result/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    
    print(message)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def print_self(message):
    # get the name of the variable that holds the message
    for name, value in globals().items():
        if value is message:
            print(f"{name} is: {value}")

