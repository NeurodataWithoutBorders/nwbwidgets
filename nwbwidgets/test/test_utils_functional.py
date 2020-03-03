from nwbwidgets.utils.functional import MemoizeMutable

class MemoizeMutableTestCase:

    def test_MemoizeMutable():
        def add_pairs():
            pass
        MemoizeMutable(add_pairs.__call__())
