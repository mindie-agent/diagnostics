"""Private directory creation helper tests."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest

from mindie_diagnostics import fallback


class PrivateDirectoryTests(unittest.TestCase):
    def test_recursive_concurrent_creation_and_existing_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = stat.S_IMODE(root.stat().st_mode)
            fallback._private_mkdir(root, exist_ok=True)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), before)
            with self.assertRaises(FileExistsError):
                fallback._private_mkdir(root)
            leaf = root / "one" / "two"
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: fallback._private_mkdir(leaf, parents=True, exist_ok=True), range(8)))
            self.assertTrue(leaf.is_dir())
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(leaf.stat().st_mode), 0o700)
            file = root / "retained"
            file.write_bytes(b"retained")
            with self.assertRaises(FileExistsError):
                fallback._private_mkdir(file, exist_ok=True)
            self.assertEqual(file.read_bytes(), b"retained")


if __name__ == "__main__":
    unittest.main()
