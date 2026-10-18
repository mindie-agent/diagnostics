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
    def assert_private_acl(self, facts):
        self.assertEqual(facts["owner"], facts["sid"])
        allowed = {facts["sid"], "S-1-5-18", "S-1-5-32-544"}
        for name, inherited in (("access", False), ("child_access", True)):
            rules = facts[name]
            self.assertTrue(all(rule["sid"] in allowed and rule["allow"] for rule in rules))
            current = [rule for rule in rules if rule["sid"] == facts["sid"]]
            self.assertEqual(len(current), 1)
            rule = current[0]
            for field, mask in (("rights", facts["full_control"]),
                                ("inheritance", facts["container_inherit"]),
                                ("inheritance", facts["object_inherit"])):
                self.assertEqual(rule[field] & mask, mask)
            self.assertEqual(rule["propagation"], 0)
            self.assertEqual(rule["inherited"], inherited)

    def test_acl_assertions_use_translated_sid_not_sddl_alias(self):
        sid = "S-1-5-21-1643835476-1616584234-1346609752-500"
        rule = dict(sid=sid, rights=2032127, inheritance=3, propagation=0,
                    inherited=False, allow=True)
        facts = dict(
            sid=sid, owner=sid, sddl="O:LAD:P(A;OICI;FA;;;LA)",
            child_sddl="O:LAD:(A;OICIID;FA;;;LA)",
            access=[rule], child_access=[dict(rule, inherited=True)],
            full_control=2032127, container_inherit=1, object_inherit=2,
        )
        self.assertNotIn(sid, facts["sddl"])
        self.assert_private_acl(facts)
        for change in ({"rights": 131209}, {"inheritance": 1}, {"inherited": False}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                altered = dict(rule, inherited=True)
                altered.update(change)
                self.assert_private_acl(dict(facts, child_access=[altered]))

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
                "$ErrorActionPreference='Stop';"
                # The CI parent may export PowerShell 7 modules to this 5.1
                # process. Read ACLs using this process's own built-in modules.
                "$env:PSModulePath=$PSHOME+'\\Modules';"
                "$s=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
                "$a=Get-Acl -LiteralPath '" + str(root).replace("'", "''") + "';"
                "$c=Get-Acl -LiteralPath '" + str(child).replace("'", "''") + "';"
                "function Rules($acl) { @($acl.Access | ForEach-Object {"
                "@{sid=$_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value;"
                "rights=[int]$_.FileSystemRights;inheritance=[int]$_.InheritanceFlags;"
                "propagation=[int]$_.PropagationFlags;inherited=$_.IsInherited;"
                "allow=($_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow)}}) };"
                "@{sid=$s;owner=$a.GetOwner([Security.Principal.SecurityIdentifier]).Value;"
                "sddl=$a.Sddl;child_sddl=$c.Sddl;access=@(Rules $a);child_access=@(Rules $c);"
                "full_control=[int][Security.AccessControl.FileSystemRights]::FullControl;"
                "container_inherit=[int][Security.AccessControl.InheritanceFlags]::ContainerInherit;"
                "object_inherit=[int][Security.AccessControl.InheritanceFlags]::ObjectInherit"
                "}|ConvertTo-Json -Compress -Depth 5"
            )
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
            facts = json.loads(subprocess.check_output(command, encoding="utf-8"))
            self.assert_private_acl(facts)
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
