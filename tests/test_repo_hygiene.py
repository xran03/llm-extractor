"""Repository hygiene: what must never be committed.

Build products and secrets do not belong in version control, and a generated
file that is also tracked will silently shadow a regenerated one. These checks
run in CI, so a mistake is caught at review time rather than after a push.
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Suffixes that are always build output.
GENERATED_SUFFIXES = (".c", ".pyd", ".so", ".o", ".obj", ".pyc")
#: Files that must never appear in this repository at all.
FORBIDDEN_NAMES = (".env",)


def tracked_files():
    result = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                            text=True, check=False)
    if result.returncode != 0:
        raise unittest.SkipTest("not a git checkout")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


class RepositoryHygieneTest(unittest.TestCase):
    def setUp(self):
        self.files = tracked_files()

    def test_no_build_products_are_tracked(self):
        offenders = [f for f in self.files if f.endswith(GENERATED_SUFFIXES)]
        self.assertEqual(offenders, [], f"generated files are tracked: {offenders}")

    def test_no_secrets_are_tracked(self):
        offenders = [f for f in self.files if Path(f).name in FORBIDDEN_NAMES]
        self.assertEqual(offenders, [], f"secret files are tracked: {offenders}")

    def test_env_example_carries_no_real_value(self):
        example = REPO / ".env.example"
        if not example.is_file():
            self.skipTest("no .env.example")
        for line in example.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip().endswith(("_KEY", "_SECRET", "_TOKEN", "_ID")):
                self.assertEqual(value.strip(), "",
                                 f"{key.strip()} must be blank in .env.example")


if __name__ == "__main__":
    unittest.main()
