import time

def timmer(func):
    def wrapper(*args, **kwargs):
        # name = input('your name>>: ').strip()
        # pwd = input('your password>>: ').strip()
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print(func.__name__, ":", stop - start)
        return res
    return wrapper