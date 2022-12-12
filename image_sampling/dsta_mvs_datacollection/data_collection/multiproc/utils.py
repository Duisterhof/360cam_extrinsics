from datetime import datetime
import re

def get_current_time_str():
    return datetime.now().strftime("%m/%d/%y %H:%M:%S.%f")

def compose_name_from_process_name(name, process_name):
    matches = re.search(r'(\d+)$', process_name)
    
    if matches is None:
        return process_name
    
    return f'{name}_{int(matches.group(1)):03d}'

def job_print_info(logger, process_name, msg):
    time_str = get_current_time_str()
    logger.info(f'{time_str} [{process_name}]: {msg}')
