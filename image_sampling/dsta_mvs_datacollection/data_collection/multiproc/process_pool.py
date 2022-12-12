
import logging
from logging.handlers import QueueHandler

# # This is required before any multiprocessing moudule is imported.
# from multiprocessing import set_start_method
# set_start_method('spawn')

import multiprocessing as mp
from multiprocessing.pool import Pool

class ReplicatedArgument(object):
    '''
    A wrapper class for a replicated argument. The user set the value and the
    total times of replication. The value has only one copy in the momory.
    
    Another implementation is to use a list constructed by [value] * count.
    '''
    def __init__(self, value, count):
        super().__init__()
        
        self.value = value
        assert isinstance(count, int), \
            f'count must be an integer, but got {type(count)}. '
        assert count > 0, \
            f'count must be greater than 0, but got {count}. '
        self.count = count
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        if index < 0 or index >= self.count:
            raise IndexError(f'index out of range. index = {index}, self.count = {self.count}. ')
        return self.value

def logger_process(logger_name: str, queue: mp.Queue, fn: str=None):
    '''
    The logger process function.
    logger_name: The name of the logger for logging.getLogger()
    queue: The multiprocessing.Queue() for receiving log messages.
    fn: Optional. The file name for saving the log. Use None to disable.
    '''
    
    # Create the logger.
    logger = logging.getLogger(logger_name)
    
    # Set handler and level.
    logger.handlers.clear()
    logger.addHandler( logging.StreamHandler() )
    # File handler.
    if fn is not None:
        logger.addHandler( logging.FileHandler(fn, mode='w') )
    logger.setLevel( logging.DEBUG )
    # print(f'logger_process: {logger.handlers}')
    
    # Run forever until receiving a None message.
    while True:
        # Query the queue for a new message.
        msg = queue.get()
        
        # Check for termination.
        if msg is None:
            # print(f'Logger process received the termination message. ')
            break
        
        # Log the message.
        logger.handle( msg )

class PoolWithLogger(object):
    '''
    A wrapper class for multiprocessing.Pool. The pool is associcated with a logger that,
    optionally, has a file handler. 
    '''
    
    def __init__(self, np, initializer, logger_name='tartanair', logger_fn=None, more_init_args=()):
        '''
        Note that the initializer must create the global variable the worker needs for logging.
        
        np: Number of processes.
        initializer: A callable that is called by each process to initialize the worker process.
        logger_name: The name of the logger for logging.getLogger().
        logger_fn: Optional. The file name for saving the log messages. Use None to disable.
        '''
        
        super().__init__()
        
        self.np             = np
        self.initializer    = initializer
        self.logger_name    = logger_name
        self.logger_fn      = logger_fn
        self.more_init_args = more_init_args
        
        # Create the queue for logging.
        self.manager   = mp.Manager()
        self.log_queue = self.manager.Queue()
        
        # Create the logger.
        self.logger = logging.getLogger(self.logger_name)
        self.logger.handlers.clear()
        self.logger.addHandler(QueueHandler(self.log_queue))
        self.logger.setLevel(logging.DEBUG)
        
        # Create and start the logger process.
        self.logger_process = mp.Process( 
                    target=logger_process, 
                    args=(self.logger_name, self.log_queue, self.logger_fn) )
        self.logger_process.start()
        
        # Internal state for the pool and result.
        self.pool = None
        self.results = None
    
    @staticmethod
    def job_prepare_logger(logger_name: str, log_queue: mp.Queue):
        '''
        A helpper function that is used in the initializer of the worker process.
        
        logger_name: The name of the logger for logging.getLogger().
        log_queue: The multiprocessing.Queue() for receiving log messages.
        '''
        
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler( QueueHandler(log_queue) )
        logger.setLevel(logging.DEBUG)
        return logger
    
    # Context manager.
    def __enter__(self):
        return self
    
    def close(self):
        '''
        Analogy to multiprocessing.Pool.close().
        '''
        if self.pool is not None:
            self.pool.close()
            
    def join(self):
        '''
        Analogy to multiprocessing.Pool.join().
        '''
        if self.pool is not None:
            self.pool.join()
    
    def close_and_join(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
    
    # Context manager.
    def __exit__(self, exc_type, exc_value, traceback):
        self.close_and_join()
        self.log_queue.put(None)
        self.logger_process.join()
        
    def map(self, func, iterable, *args, **kwargs):
        '''
        Analogy to multiprocessing.Pool.map(). However, starmap_async() is used internally.
        '''
        
        # with Pool(self.np, initializer=self.initializer, initargs=(self.logger_name, self.log_queue,)) as pool:
        #     # Using async version.
        #     # https://stackoverflow.com/questions/60094970/multiprocessing-map-async-hangs-but-multiprocessing-map-works-correctly
        #     results = pool.starmap_async(func, iterable, *args, **kwargs)
            
        #     try:
        #         self.results = [ res for res in results.get() ]
        #     except Exception as exc:
        #         print(exc)
        
        # First clear the pool.
        self.close_and_join()
        
        # Create the pool.
        self.pool = Pool(self.np, initializer=self.initializer, initargs=(self.logger_name, self.log_queue, *self.more_init_args))
        results = self.pool.starmap_async(func, iterable, *args, **kwargs)
        self.results = results.get()
        
        return self.results
