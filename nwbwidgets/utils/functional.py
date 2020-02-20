import pickle


class MemoizeMutable:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args, **kwds):
        this_str = pickle.dumps(args, 1) + pickle.dumps(kwds, 1)
        if this_str not in self.memo:
            self.memo[this_str] = self.fn(*args, **kwds)
        return self.memo[this_str]
