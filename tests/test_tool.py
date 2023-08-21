import unittest
import logging
import os
import json

logger = logging.getLogger()


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_list(self):
        a = None
        self.assertTrue(not a)
        a = []
        self.assertTrue(not a)

    def test_md5(self):
        from utils import toolutil
        s = 'abc'
        smd = toolutil.get_md5(s)
        self.assertEqual(smd, '900150983cd24fb0d6963f7d28e17f72')

    def test_file_path(self):
        f = os.path.abspath(__file__)
        print(f)
        print(os.path.basename(f))

    def test_json(self):
        j_s = '{"a":3}'
        jo = json.loads(j_s)
        print(type(jo))
        self.assertIsInstance(jo, dict)

    def test_chinese(self):
        from utils import toolutil
        s = 'hello'
        c = toolutil.check_contain_chinese(s)
        self.assertEqual(c, False)

        s = '\u4e00 hello  world !'
        c = toolutil.check_contain_chinese(s)
        self.assertEqual(c, True)

    def test_read_conf(self):
        from utils import fileutil
        file_path = 'config.ini'
        conf = fileutil.read_dict_conf(file_path)
        print(conf)
        self.assertTrue(3>2)



if __name__ == '__main__':
    unittest.main()
