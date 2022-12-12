
from datetime import datetime
import numpy as np

from process_pool import ( ReplicatedArgument, PoolWithLogger )

def job_initializer(logger_name, log_queue):
    # Use global variables to transfer variables to the job process.
    # This funciton is called in the job process.
    # https://superfastpython.com/multiprocessing-pool-initializer/
    global P_JOB_LOGGER
    
    # The logger.
    P_JOB_LOGGER = PoolWithLogger.job_prepare_logger(logger_name, log_queue)
    
    # print(P_JOB_LOGGER.handlers)

def job(data, arg1, arg2):
    global P_JOB_LOGGER
    
    # if arg1 == 4:
    #     raise Exception('arg1 == 4')
    
    time_str = datetime.now().strftime("%m/%d/%y %H:%M:%S.%f")
    
    P_JOB_LOGGER.info(
        f'{time_str}, arg1 = {arg1}, data = {data}, arg2 = {arg2}')

    return dict(arg1=arg1, arg2=arg2)

if __name__ == '__main__':
    N = 10
    
    # The raw data.
    data = np.random.rand((N)).astype(np.float32)
    print(f'data is \n {data}')

    # The list argument.
    arg1_list = list(range(N))
    
    # The replicated argument.
    arg2 = ReplicatedArgument('hello', N)
    
    # The pool.
    zipped = zip(data, arg1_list, arg2)
    with PoolWithLogger(4, job_initializer, 'tartanair', './logger_output.log') as pool:
        results = pool.map( job, zipped )
        print(f'Main: \n{results}')
    
    print('Main: Done.')
