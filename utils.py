import os
import signal

__all__ = ['AttrDict']


def _term(sig_num, addition):
    print('current pid is %s, group id is %s' % (os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _term)
signal.signal(signal.SIGINT, _term)


class AttrDict(dict):
    def __getattr__(self, key):#对没有定义的属性名和实例,会用属性名作为字符串调用这个方法
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value
