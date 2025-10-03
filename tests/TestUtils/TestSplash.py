import unittest
import os
from msentity.core import *
from msentity.io.msp import read_msp
from msentity.utils.splash import *

class TestSplash(unittest.TestCase):
    def setUp(self):
        # --- Create a dummy MSP file ---
        self.test_file = os.path.join("tests", "TestMassEntityIO", "dummy_files", "test_dummy.msp")
        self.ds = read_msp(self.test_file)

    def test_add_splash_to_dataset(self):
        new_ds = add_splash_to_dataset(self.ds, splash_column="SPLASH_ID", in_place=True)
        self.assertIn("SPLASH_ID", new_ds.columns)
        self.assertEqual(len(new_ds), len(self.ds))
        self.assertTrue(all(new_ds["SPLASH_ID"].str.startswith("splash")))

if __name__ == "__main__":
    unittest.main()
