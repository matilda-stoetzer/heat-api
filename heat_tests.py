# tester

import unittest
import numpy as np

class Tests(unittest.TestCase):
    def test(self):
        v = np.array([1, 1])
        w = np.array([1, 1])
        self.assertEqual(v, w)

unittest.main()