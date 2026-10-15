"""Private creation preserves both elevated and ordinary Windows user access."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mindie_diagnostics import fallback


class PrivateDirectoryTests(unittest.TestCase):
    def test_existing_directory_is_not_repermissioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = stat.S_IMODE(root.stat().st_mode)
            fallback._private_mkdir(root, exist_ok=True)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), before)
            with self.assertRaises(FileExistsError):
                fallback._private_mkdir(root)

    def test_recursive_concurrent_creation_and_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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

    def test_creation_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "missing"
            owner = fallback if os.name == "nt" else os
            name = "_windows_private_mkdir" if os.name == "nt" else "mkdir"
            with patch.object(owner, name, side_effect=PermissionError("synthetic create denied")):
                with self.assertRaises(PermissionError):
                    fallback._private_mkdir(target)
            self.assertFalse(target.exists())

    def test_fallback_remains_standalone(self):
        # The adapter vendors one file; its private-directory helper must not
        # import another diagnostics module to create a local directory.
        spec = importlib.util.spec_from_file_location("standalone_diagnostic_fixture", fallback.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            module._private_mkdir(Path(temporary) / "private")

    @unittest.skipUnless(os.name == "nt", "native Windows directory ACL")
    def test_native_user_owner_and_inheritance_without_changing_existing_acl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private"
            fallback._private_mkdir(root)
            child = root / "child"
            child.mkdir()
            (child / "readback").write_bytes(b"synthetic")
            script = (
                "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
                "$s=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
                "$a=Get-Acl -LiteralPath '" + str(root).replace("'", "''") + "';"
                "$c=Get-Acl -LiteralPath '" + str(child).replace("'", "''") + "';"
                "@{sid=$s;owner=$a.GetOwner([Security.Principal.SecurityIdentifier]).Value;"
                "sddl=$a.Sddl;child_sddl=$c.Sddl}|ConvertTo-Json -Compress"
            )
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
            facts = json.loads(subprocess.check_output(command, encoding="utf-8"))
            self.assertEqual(facts["owner"], facts["sid"])
            self.assertIn("(A;OICI;FA;;;" + facts["sid"] + ")", facts["sddl"])
            self.assertIn(facts["sid"], facts["child_sddl"])
            for principal in ("WD", "BU", "AU"):
                self.assertNotIn(";;;" + principal + ")", facts["sddl"])
            fallback._private_mkdir(root, exist_ok=True)
            self.assertEqual(json.loads(subprocess.check_output(command, encoding="utf-8"))["sddl"], facts["sddl"])
            print("NATIVE_PRIVATE_DIRECTORY=" + json.dumps(facts), flush=True)

    @unittest.skipUnless(os.name == "nt", "native Windows junction")
    def test_native_junction_is_not_followed_or_repermissioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "junction"
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, check=True)
            try:
                with self.assertRaises(OSError):
                    fallback._private_mkdir(link, exist_ok=True)
                with self.assertRaises(OSError):
                    fallback._private_mkdir(link / "child", parents=True, exist_ok=True)
                self.assertFalse((target / "child").exists())
            finally:
                link.rmdir()


if __name__ == "__main__":
    unittest.main()
