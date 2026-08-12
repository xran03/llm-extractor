"""CLI argument handling and credential resolution (env / .env / paste)."""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from llm_extractor import cli
from llm_extractor.credentials import (PROMPT_SENTINEL, find_env_file, load_env_file,
                                       mask, resolve_secret)
from llm_extractor.settings import build_settings


class ArgumentParsingTest(unittest.TestCase):
    def test_run_is_the_default_subcommand(self):
        args = cli.build_parser().parse_args(["run", "-i", "docs", "-o", "out"])
        self.assertEqual(args.cmd, "run")
        self.assertEqual(args.input, "docs")
        self.assertEqual(args.output, "out")

    def test_short_flags_match_the_documented_invocation(self):
        args = cli.build_parser().parse_args(
            ["run", "-i", "folder", "-o", "output", "--api", "aimodelhub"])
        self.assertEqual(args.api, "aimodelhub")

    def test_long_flags(self):
        args = cli.build_parser().parse_args(
            ["run", "--input", "f", "--output", "o", "--template", "immunogenicity"])
        self.assertEqual(args.template, "immunogenicity")

    def test_ocr_policy_choices(self):
        args = cli.build_parser().parse_args(["run", "-i", "f", "--ocr", "always"])
        self.assertEqual(args.ocr, "always")

    def test_audit_defaults(self):
        args = cli.build_parser().parse_args(["audit"])
        self.assertEqual(args.n, 20)
        self.assertEqual(args.strategy, "random")

    def test_serve_defaults(self):
        args = cli.build_parser().parse_args(["serve"])
        self.assertEqual(args.port, 8080)
        self.assertEqual(args.host, "127.0.0.1")

    def test_every_subcommand_is_routed(self):
        self.assertEqual(sorted(cli.COMMANDS), sorted(cli.SUBCOMMANDS))


class ParamParsingTest(unittest.TestCase):
    def test_simple_pairs(self):
        self.assertEqual(cli.parse_params(["a=1", "b=text"]), {"a": 1, "b": "text"})

    def test_json_values_are_decoded(self):
        params = cli.parse_params(['fields=["a","b"]', "flag=true"])
        self.assertEqual(params["fields"], ["a", "b"])
        self.assertIs(params["flag"], True)

    def test_value_with_equals_signs_is_preserved(self):
        self.assertEqual(cli.parse_params(["q=a=b"])["q"], "a=b")

    def test_empty_and_malformed_are_ignored(self):
        self.assertEqual(cli.parse_params(["", "=x"]), {})


class DefaultSubcommandTest(unittest.TestCase):
    def test_bare_flags_get_run_prepended(self):
        argv = ["-i", "docs", "-o", "out"]
        parsed = cli.build_parser().parse_args(["run", *argv])
        self.assertEqual(parsed.cmd, "run")

    def test_named_subcommand_is_left_alone(self):
        self.assertEqual(cli.build_parser().parse_args(["cache", "stats"]).cmd, "cache")


class CommandOutputTest(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def test_sources_lists_connectors(self):
        code, output = self._run(["sources"])
        self.assertEqual(code, 0)
        self.assertIn("folder", output)
        self.assertIn("patents", output)

    def test_formats_lists_every_supported_format(self):
        code, output = self._run(["formats"])
        self.assertEqual(code, 0)
        for name in ("pdf", "docx", "pptx", "xlsx", "csv", "png", "jpeg", "eml"):
            self.assertIn(name, output)
        self.assertIn("detected from content", output)

    def test_templates_lists_builtins(self):
        code, output = self._run(["templates"])
        self.assertEqual(code, 0)
        self.assertIn("generic", output)
        self.assertIn("immunogenicity", output)

    def test_templates_show_prints_the_schema(self):
        code, output = self._run(["templates", "--show", "generic"])
        self.assertEqual(code, 0)
        self.assertIn("json_schema", output)

    def test_run_without_input_is_a_usage_error(self):
        self.assertEqual(cli.main(["run", "--api", "llmhub"]), 2)

    def test_no_command_ever_prompts_implicitly(self):
        """A missing key must fail fast, never block a script waiting on stdin."""
        called = []

        def tripwire(label):
            called.append(label)
            return "should-not-be-used"

        import llm_extractor.credentials as credentials

        original = credentials.prompt_secret
        credentials.prompt_secret = tripwire
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for argv in (["check", "--cache-dir", tmp],
                             ["cache", "stats", "--cache-dir", tmp],
                             ["run", "--api", "llmhub"]):
                    self._run(argv)
        finally:
            credentials.prompt_secret = original
        self.assertEqual(called, [])

    def test_dash_api_key_is_the_documented_paste_path(self):
        parser = cli.build_parser()
        args = parser.parse_args(["run", "-i", "docs", "--api-key", "-"])
        self.assertEqual(args.api_key, cli.PROMPT_SENTINEL)

    def test_unknown_template_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(["run", "-i", tmp, "--template", "not-a-template"])
        self.assertEqual(code, 2)

    def test_cache_stats_on_an_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, output = self._run(["cache", "stats", "--cache-dir",
                                      str(Path(tmp) / "c")])
        self.assertEqual(code, 0)
        self.assertIn("entries", output)


class CredentialsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Resolved: Windows hands out 8.3 short names (RUNNER~1) for temp dirs,
        # and path lookups return the long form, so unresolved paths compare unequal.
        self.dir = Path(self.tmp.name).resolve()
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("UNIT_TEST_")}

    def tearDown(self):
        for key in [k for k in os.environ if k.startswith("UNIT_TEST_")]:
            os.environ.pop(key)
        os.environ.update(self._saved)
        self.tmp.cleanup()

    def test_cli_value_wins(self):
        os.environ["UNIT_TEST_KEY"] = "from-env"
        self.assertEqual(resolve_secret("from-cli", ["UNIT_TEST_KEY"], "k"), "from-cli")

    def test_environment_is_used_when_no_cli_value(self):
        os.environ["UNIT_TEST_KEY"] = "from-env"
        self.assertEqual(resolve_secret(None, ["UNIT_TEST_KEY"], "k"), "from-env")

    def test_first_matching_env_name_wins(self):
        os.environ["UNIT_TEST_SECOND"] = "second"
        self.assertEqual(
            resolve_secret(None, ["UNIT_TEST_FIRST", "UNIT_TEST_SECOND"], "k"), "second")

    def test_dash_forces_the_paste_prompt(self):
        value = resolve_secret(PROMPT_SENTINEL, ["UNIT_TEST_KEY"], "k",
                               prompter=lambda label: "pasted")
        self.assertEqual(value, "pasted")

    def test_prompt_is_used_as_a_last_resort(self):
        value = resolve_secret(None, ["UNIT_TEST_ABSENT"], "k", allow_prompt=True,
                               prompter=lambda label: "pasted")
        self.assertEqual(value, "pasted")

    def test_no_value_and_no_prompt_returns_empty(self):
        self.assertEqual(resolve_secret(None, ["UNIT_TEST_ABSENT"], "k"), "")

    def test_env_file_is_parsed(self):
        env = self.dir / ".env"
        env.write_text('UNIT_TEST_A=plain\nUNIT_TEST_B="quoted"\n'
                       "export UNIT_TEST_C=exported\n# comment\n\n", encoding="utf-8")
        values = load_env_file(env)
        self.assertEqual(values["UNIT_TEST_A"], "plain")
        self.assertEqual(values["UNIT_TEST_B"], "quoted")
        self.assertEqual(values["UNIT_TEST_C"], "exported")

    def test_env_file_discovery_walks_upward(self):
        nested = self.dir / "a" / "b"
        nested.mkdir(parents=True)
        (self.dir / ".env").write_text("UNIT_TEST_X=1", encoding="utf-8")
        found = find_env_file(nested)
        self.assertIsNotNone(found)
        self.assertEqual(found.resolve(), (self.dir / ".env").resolve())

    def test_real_environment_beats_the_env_file(self):
        os.environ["UNIT_TEST_D"] = "from-real-env"
        env = self.dir / ".env"
        env.write_text("UNIT_TEST_D=from-file", encoding="utf-8")
        load_env_file(env)
        self.assertEqual(resolve_secret(None, ["UNIT_TEST_D"], "k"), "from-real-env")

    def test_mask_never_reveals_the_secret(self):
        masked = mask("supersecretvalue")
        self.assertNotIn("secretvalue", masked)
        self.assertIn("len=16", masked)
        self.assertEqual(mask(""), "(not set)")


class SettingsTest(unittest.TestCase):
    def tearDown(self):
        for key in ("LLM_HUB_BASE_URL", "LLM_HUB_API_KEY", "AI_MODEL_HUB_BASE_URL",
                    "AI_MODEL_HUB_API_KEY", "LLM_EXTRACTOR_API"):
            os.environ.pop(key, None)

    def test_backend_selects_its_own_env_prefix(self):
        os.environ["LLM_HUB_BASE_URL"] = "https://one"
        os.environ["AI_MODEL_HUB_BASE_URL"] = "https://two"
        self.assertEqual(build_settings("llmhub").base_url, "https://one")
        self.assertEqual(build_settings("aimodelhub").base_url, "https://two")

    def test_cli_overrides_environment(self):
        os.environ["LLM_HUB_BASE_URL"] = "https://one"
        self.assertEqual(build_settings("llmhub", base_url="https://cli").base_url,
                         "https://cli")

    def test_default_api_can_come_from_environment(self):
        os.environ["LLM_EXTRACTOR_API"] = "aimodelhub"
        self.assertEqual(build_settings().api, "aimodelhub")

    def test_ocr_and_agent_models_default_to_the_main_model(self):
        settings = build_settings("llmhub", model="my-model")
        self.assertEqual(settings.ocr_model, "my-model")
        self.assertEqual(settings.agent_model, "my-model")

    def test_describe_masks_secrets(self):
        described = build_settings("llmhub", api_key="supersecret").describe()
        self.assertNotIn("supersecret", str(described))

    def test_unknown_overrides_are_ignored(self):
        settings = build_settings("llmhub", not_a_field="x")
        self.assertFalse(hasattr(settings, "not_a_field"))


if __name__ == "__main__":
    unittest.main()
