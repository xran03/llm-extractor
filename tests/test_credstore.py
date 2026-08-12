"""Saved credential store: paste once, run many times."""
from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_extractor import cli
from llm_extractor import credstore
from llm_extractor import credentials
from llm_extractor.credentials import resolve_secret
from llm_extractor.settings import build_settings


class StoreTestCase(unittest.TestCase):
    """Redirect the store into a temporary directory for every test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._previous = os.environ.get(credstore.CONFIG_DIR_ENV)
        os.environ[credstore.CONFIG_DIR_ENV] = self._tmp.name
        # A real key in the ambient environment would mask what we assert on.
        self._saved_env = {}
        for name in ("LLM_HUB_API_KEY", "LLM_HUB_KEY", "LLM_HUB_BASE_URL",
                     "AI_MODEL_HUB_API_KEY", "AI_MODEL_HUB_KEY",
                     "AI_MODEL_HUB_BASE_URL", "LLM_EXTRACTOR_API"):
            self._saved_env[name] = os.environ.pop(name, None)
        # Neutralise any developer's real .env: mark it loaded but empty, so
        # `get_env` neither reads the file nor finds a value in the cache.
        self._saved_cache = dict(credentials._ENV_CACHE)
        self._saved_origin = list(credentials._ENV_LOADED_FROM)
        credentials._ENV_CACHE.clear()
        credentials._ENV_LOADED_FROM[:] = [str(Path(self._tmp.name) / ".env")]

    def tearDown(self):
        if self._previous is None:
            os.environ.pop(credstore.CONFIG_DIR_ENV, None)
        else:
            os.environ[credstore.CONFIG_DIR_ENV] = self._previous
        for name, value in self._saved_env.items():
            if value is not None:
                os.environ[name] = value
        credentials._ENV_CACHE.clear()
        credentials._ENV_CACHE.update(self._saved_cache)
        credentials._ENV_LOADED_FROM[:] = self._saved_origin
        self._tmp.cleanup()


class RoundTripTest(StoreTestCase):
    def test_missing_store_reads_as_empty(self):
        self.assertEqual(credstore.load_store(), {})
        self.assertEqual(credstore.stored_api_key("llmhub"), "")
        self.assertEqual(credstore.stored_base_url("llmhub"), "")

    def test_save_then_read(self):
        credstore.save_credentials("llmhub", api_key="k-123", base_url="https://gw")
        self.assertEqual(credstore.stored_api_key("llmhub"), "k-123")
        self.assertEqual(credstore.stored_base_url("llmhub"), "https://gw")

    def test_trailing_slash_is_normalised(self):
        credstore.save_credentials("llmhub", api_key="k", base_url="https://gw/")
        self.assertEqual(credstore.stored_base_url("llmhub"), "https://gw")

    def test_backends_are_independent(self):
        credstore.save_credentials("llmhub", api_key="one", base_url="https://a")
        credstore.save_credentials("aimodelhub", api_key="two", base_url="https://b")
        self.assertEqual(credstore.stored_api_key("llmhub"), "one")
        self.assertEqual(credstore.stored_api_key("aimodelhub"), "two")

    def test_saving_one_field_keeps_the_other(self):
        credstore.save_credentials("llmhub", api_key="k", base_url="https://gw")
        credstore.save_credentials("llmhub", api_key="rotated")
        self.assertEqual(credstore.stored_api_key("llmhub"), "rotated")
        self.assertEqual(credstore.stored_base_url("llmhub"), "https://gw")

    def test_corrupt_store_is_treated_as_absent(self):
        credstore.store_path().parent.mkdir(parents=True, exist_ok=True)
        credstore.store_path().write_text("{not json", encoding="utf-8")
        self.assertEqual(credstore.load_store(), {})
        self.assertEqual(credstore.stored_api_key("llmhub"), "")

    def test_delete_one_backend_leaves_the_other(self):
        credstore.save_credentials("llmhub", api_key="one")
        credstore.save_credentials("aimodelhub", api_key="two")
        self.assertTrue(credstore.delete_credentials("llmhub"))
        self.assertEqual(credstore.stored_api_key("llmhub"), "")
        self.assertEqual(credstore.stored_api_key("aimodelhub"), "two")

    def test_delete_unknown_backend_reports_nothing_removed(self):
        self.assertFalse(credstore.delete_credentials("llmhub"))

    def test_delete_everything_removes_the_file(self):
        credstore.save_credentials("llmhub", api_key="one")
        self.assertTrue(credstore.delete_credentials())
        self.assertFalse(credstore.store_path().exists())

    def test_removing_the_last_entry_removes_the_file(self):
        credstore.save_credentials("llmhub", api_key="one")
        credstore.delete_credentials("llmhub")
        self.assertFalse(credstore.store_path().exists())


class PermissionTest(StoreTestCase):
    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_secret_is_owner_only(self):
        path = credstore.save_credentials("llmhub", api_key="secret")
        mode = stat.S_IMODE(path.stat().st_mode)
        self.assertEqual(mode, 0o600, f"world-readable secret: {oct(mode)}")

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_directory_is_owner_only(self):
        path = credstore.save_credentials("llmhub", api_key="secret")
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        self.assertEqual(mode, 0o700, f"world-readable directory: {oct(mode)}")

    def test_no_temporary_file_is_left_behind(self):
        credstore.save_credentials("llmhub", api_key="secret")
        leftovers = list(credstore.config_dir().glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_stored_json_is_readable_by_the_owner(self):
        credstore.save_credentials("llmhub", api_key="secret", base_url="https://gw")
        data = json.loads(credstore.store_path().read_text(encoding="utf-8"))
        self.assertEqual(data["llmhub"]["api_key"], "secret")


class PrecedenceTest(StoreTestCase):
    """An explicit flag, the environment and .env all outrank the store."""

    def test_store_is_used_when_nothing_else_is_set(self):
        credstore.save_credentials("llmhub", api_key="stored-key",
                                   base_url="https://stored")
        settings = build_settings("llmhub")
        self.assertEqual(settings.api_key, "stored-key")
        self.assertEqual(settings.base_url, "https://stored")

    def test_cli_value_beats_the_store(self):
        credstore.save_credentials("llmhub", api_key="stored-key")
        settings = build_settings("llmhub", api_key="flag-key", base_url="https://x")
        self.assertEqual(settings.api_key, "flag-key")

    def test_environment_beats_the_store(self):
        credstore.save_credentials("llmhub", api_key="stored-key")
        os.environ["LLM_HUB_API_KEY"] = "env-key"
        try:
            settings = build_settings("llmhub", base_url="https://x")
            self.assertEqual(settings.api_key, "env-key")
        finally:
            os.environ.pop("LLM_HUB_API_KEY", None)

    def test_cli_base_url_beats_the_store(self):
        credstore.save_credentials("llmhub", base_url="https://stored")
        settings = build_settings("llmhub", api_key="k", base_url="https://flag")
        self.assertEqual(settings.base_url, "https://flag")

    def test_store_does_not_leak_across_backends(self):
        credstore.save_credentials("llmhub", api_key="only-llmhub",
                                   base_url="https://a")
        settings = build_settings("aimodelhub")
        self.assertEqual(settings.api_key, "")

    def test_fallback_is_skipped_when_the_environment_wins(self):
        calls = []

        def fallback():
            calls.append(1)
            return "stored"

        os.environ["LLM_HUB_API_KEY"] = "env-key"
        try:
            value = resolve_secret(None, ["LLM_HUB_API_KEY"], "key", fallback=fallback)
        finally:
            os.environ.pop("LLM_HUB_API_KEY", None)
        self.assertEqual(value, "env-key")
        self.assertEqual(calls, [], "the store was consulted unnecessarily")

    def test_prompt_is_not_reached_when_the_store_has_a_key(self):
        def prompter(label):
            raise AssertionError("prompted despite a saved key")

        value = resolve_secret(None, ["MISSING_VAR"], "key", allow_prompt=True,
                               prompter=prompter, fallback=lambda: "stored")
        self.assertEqual(value, "stored")


class DescribeTest(StoreTestCase):
    def test_describes_an_empty_store(self):
        self.assertIn("login", credstore.describe_store())

    def test_describes_saved_backends_without_revealing_the_key(self):
        credstore.save_credentials("llmhub", api_key="super-secret-value")
        description = credstore.describe_store()
        self.assertIn("llmhub", description)
        self.assertNotIn("super-secret-value", description)


class CliLoginTest(StoreTestCase):
    """`login` / `logout` behaviour, without touching the network."""

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def test_login_and_logout_are_routed(self):
        self.assertEqual(sorted(cli.COMMANDS), sorted(cli.SUBCOMMANDS))

    def test_login_saves_without_verification(self):
        code, _ = self._run(["login", "--api", "llmhub", "--base-url", "https://gw",
                             "--api-key", "pasted-key", "--no-verify"])
        self.assertEqual(code, 0)
        self.assertEqual(credstore.stored_api_key("llmhub"), "pasted-key")
        self.assertEqual(credstore.stored_base_url("llmhub"), "https://gw")

    def test_login_never_prints_the_key(self):
        _, output = self._run(["login", "--api", "llmhub", "--base-url", "https://gw",
                               "--api-key", "top-secret-value", "--no-verify"])
        self.assertNotIn("top-secret-value", output)

    def test_login_without_a_terminal_or_key_is_a_usage_error(self):
        code, output = self._run(["login", "--api", "llmhub", "--base-url", "https://gw"])
        self.assertEqual(code, 2)
        self.assertIn("api-key", output)
        self.assertFalse(credstore.store_path().exists())

    def test_login_without_a_base_url_is_a_usage_error(self):
        code, output = self._run(["login", "--api", "llmhub", "--api-key", "k",
                                  "--no-verify"])
        self.assertEqual(code, 2)
        self.assertIn("base URL", output)

    def test_login_reuses_a_base_url_already_saved(self):
        credstore.save_credentials("llmhub", base_url="https://saved")
        code, _ = self._run(["login", "--api", "llmhub", "--api-key", "k", "--no-verify"])
        self.assertEqual(code, 0)
        self.assertEqual(credstore.stored_base_url("llmhub"), "https://saved")

    def test_logout_removes_the_saved_key(self):
        self._run(["login", "--api", "llmhub", "--base-url", "https://gw",
                   "--api-key", "k", "--no-verify"])
        code, _ = self._run(["logout", "--api", "llmhub"])
        self.assertEqual(code, 0)
        self.assertEqual(credstore.stored_api_key("llmhub"), "")

    def test_logout_all_clears_every_backend(self):
        credstore.save_credentials("llmhub", api_key="a")
        credstore.save_credentials("aimodelhub", api_key="b")
        code, _ = self._run(["logout", "--all"])
        self.assertEqual(code, 0)
        self.assertFalse(credstore.store_path().exists())

    def test_logout_on_an_empty_store_is_not_an_error(self):
        code, _ = self._run(["logout", "--api", "llmhub"])
        self.assertEqual(code, 0)

    def test_check_reports_the_saved_key_without_revealing_it(self):
        self._run(["login", "--api", "llmhub", "--base-url", "https://gw",
                   "--api-key", "top-secret-value", "--no-verify"])
        _, output = self._run(["check", "--api", "llmhub"])
        self.assertIn("saved key", output)
        self.assertNotIn("top-secret-value", output)


class ImplicitPromptTest(StoreTestCase):
    """The paste prompt must never block a script, CI job or pipe."""

    def test_not_interactive_when_stdin_is_not_a_terminal(self):
        self.assertFalse(cli._interactive())

    def test_run_does_not_prompt_without_a_terminal(self):
        calls = []

        def tripwire(label):
            calls.append(label)
            return "should-not-be-used"

        original = credentials.prompt_secret
        credentials.prompt_secret = tripwire
        try:
            with tempfile.TemporaryDirectory() as docs:
                buffer = io.StringIO()
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    cli.main(["run", "-i", docs, "--api", "llmhub",
                              "--cache-dir", str(Path(docs) / "c")])
        finally:
            credentials.prompt_secret = original
        self.assertEqual(calls, [])


class VerificationTest(StoreTestCase):
    """A key is only "verified" when authentication was actually exercised."""

    class _Provider:
        def __init__(self, models=None, complete_error=None):
            self._models = models if models is not None else ["m1"]
            self._complete_error = complete_error
            self.completed = False

        def list_models(self):
            if isinstance(self._models, Exception):
                raise self._models
            return self._models

        def complete(self, *args, **kwargs):
            self.completed = True
            if self._complete_error:
                raise self._complete_error
            return object()

    def _verify_with(self, provider):
        import llm_extractor.providers as providers

        original = providers.build_provider
        providers.build_provider = lambda *a, **k: provider
        try:
            return cli.verify_credentials(
                build_settings("llmhub", base_url="https://gw", api_key="k",
                               cache_enabled=False))
        finally:
            providers.build_provider = original

    def test_open_models_endpoint_alone_is_not_proof(self):
        """Regression: some gateways serve /v1/models without authentication."""
        from llm_extractor.providers import ProviderError

        provider = self._Provider(
            models=["m1", "m2"],
            complete_error=ProviderError("llmhub HTTP 401 from /v1/chat/completions: "
                                         "{\"error\": \"Invalid or expired key\"}"))
        ok, detail = self._verify_with(provider)
        self.assertFalse(ok, "a rejected key was reported as verified")
        self.assertIn("rejected", detail)
        self.assertTrue(provider.completed, "authentication was never exercised")

    def test_working_key_is_verified(self):
        provider = self._Provider(models=["m1", "m2"])
        ok, detail = self._verify_with(provider)
        self.assertTrue(ok)
        self.assertIn("test call accepted", detail)

    def test_unreachable_gateway_is_reported(self):
        from llm_extractor.providers import ProviderError

        provider = self._Provider(models=ProviderError("network error"))
        ok, detail = self._verify_with(provider)
        self.assertFalse(ok)
        self.assertIn("could not be reached", detail)

    def test_non_auth_failure_does_not_veto_the_key(self):
        """An unavailable model is not evidence that the key is wrong."""
        from llm_extractor.providers import ProviderError

        provider = self._Provider(
            models=["m1"], complete_error=ProviderError("llmhub HTTP 400: unknown model"))
        ok, detail = self._verify_with(provider)
        self.assertTrue(ok)
        self.assertIn("not be fully exercised", detail)


if __name__ == "__main__":
    unittest.main()
