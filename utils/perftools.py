import concurrent
from concurrent.futures import ProcessPoolExecutor
import time
from functools import wraps


def timefn(fn):
    """
    修饰器：自动计算函数调用耗时时间
    fn：函数名
    使用时，直接在函数名上 @timefn
    """

    @wraps(fn)
    def measure_time(*args, **kwargs):
        """
        args: 函数 fn 的参数
        kwargs：函数 fn 的参数 ，形式是 key-value
        """
        t1 = time.time()
        # 函数调用有返回值
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn:", fn.__name__, " took ", str(t2 - t1), " seconds")
        return result

    return measure_time


def parallel_run_mapping(func, kv, max_workers=5):
    '''
    TODO : code : from wejoy_analysis
    '''
    result = dict()
    i = 0
    I = len(kv)
    step_show = min(I // 20 + 1, 1000)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_res_d = {executor.submit(func, v): k
                        for k, v in kv.items()}
        for fut in concurrent.futures.as_completed(future_res_d):
            name = future_res_d[fut]
            result[name] = fut.result()
            if i % step_show == 0:
                print(f'[{i} / {I}] done.')
            i += 1
    return result
