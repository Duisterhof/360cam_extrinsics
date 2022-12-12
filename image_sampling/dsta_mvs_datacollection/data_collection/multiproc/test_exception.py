
from multiprocessing.pool import Pool

class DummyPool(Pool):
    def __exit__(self, exc_type, exc_value, traceback):
        print('DummyPool.__exit__ get called. ')
        return super().__exit__(exc_type, exc_value, traceback)
    
def dummy_func(*args):
    print('dummy_func get called. ')
    raise Exception('Dummy exception.')

if __name__ == '__main__':
    try:
        with DummyPool(2) as pool:
            pool.map(dummy_func, range(2))
    except Exception as exc:
        print(f'Main process catch exception: {exc}')
    
    print('Cleanly exit the context manager. ')
