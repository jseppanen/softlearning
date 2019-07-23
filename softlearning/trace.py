
import inspect

tracefile = open('repro.log', 'w')


def TRACE(*args, **kvs):
    assert not args
    # inspect call site
    frame = inspect.currentframe().f_back
    try:
        filename, lineno, func_name, _, _ = inspect.getframeinfo(frame)
    finally:
        # prevent gc cycle
        del frame

    for name in kvs:
        tracefile.write(f'{filename}:{lineno}:{func_name}: {name} =\n')
        tracefile.write(f'{repr(kvs[name])}\n')
    tracefile.flush()
