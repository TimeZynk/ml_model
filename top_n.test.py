import unittest
from top_n import top_n

class TestTopN(unittest.TestCase):
    def test_top_n(self):
        testList = list(range(100))
        self.assertEqual(top_n(testList, 5), [99, 98, 97, 96, 95])

if __name__ == '__main__':
    unittest.main()