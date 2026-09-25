import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_research_agent.cli import build_parser, build_runner


class TestBuildParser(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.prompt)
        self.assertEqual(args.max_iterations, 5)
        self.assertEqual(args.model, "nex-agi/nex-n2.5-mini:free")
        self.assertEqual(args.search_provider, "tavily")
        self.assertFalse(args.tui)

    def test_tui_flag(self):
        args = build_parser().parse_args(["question here", "--tui"])
        self.assertEqual(args.prompt, "question here")
        self.assertTrue(args.tui)

    def test_custom_options(self):
        args = build_parser().parse_args(
            ["p", "--max-iterations", "3", "--model", "m", "--search-provider", "exa"]
        )
        self.assertEqual(args.max_iterations, 3)
        self.assertEqual(args.model, "m")
        self.assertEqual(args.search_provider, "exa")

    def test_invalid_search_provider_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--search-provider", "nope"])

    def test_non_positive_max_iterations_errors(self):
        parser = build_parser()
        args = parser.parse_args(["--max-iterations", "0"])
        self.assertEqual(args.max_iterations, 0)
        with (
            patch("deep_research_agent.cli.load_environment"),
            patch("deep_research_agent.cli.ResearchAgentStorage") as storage_cls,
            self.assertRaises(SystemExit),
        ):
            from deep_research_agent.cli import main

            main(["--max-iterations", "0"])

        storage_cls.assert_not_called()


class TestBuildRunner(unittest.TestCase):
    @patch("deep_research_agent.cli.OpenRouterLLMProvider")
    @patch("deep_research_agent.cli.SearchService")
    def test_default_search_provider_is_tavily(
        self, search_service: MagicMock, llm: MagicMock
    ):
        tavily_cls = MagicMock()
        exa_cls = MagicMock()
        with patch(
            "deep_research_agent.cli.SEARCH_PROVIDERS",
            {
                "tavily": tavily_cls,
                "exa": exa_cls,
            },
        ):
            runner = build_runner(max_iterations=4)
        tavily_cls.assert_called_once_with()
        exa_cls.assert_not_called()
        search_service.assert_called_once()
        llm.assert_called_once_with(model_name="nex-agi/nex-n2.5-mini:free")
        self.assertIs(runner.events, None)
        self.assertIsNone(runner.storage)

    @patch("deep_research_agent.cli.OpenRouterLLMProvider")
    @patch("deep_research_agent.cli.SearchService")
    def test_uses_injected_storage(self, search_service: MagicMock, llm: MagicMock):
        storage = MagicMock()
        with patch(
            "deep_research_agent.cli.SEARCH_PROVIDERS",
            {"tavily": MagicMock(), "exa": MagicMock()},
        ):
            runner = build_runner(max_iterations=2, storage=storage)

        self.assertIs(runner.storage, storage)

    @patch("deep_research_agent.cli.OpenRouterLLMProvider")
    @patch("deep_research_agent.cli.SearchService")
    def test_exa_search_provider(self, search_service: MagicMock, llm: MagicMock):
        tavily_cls = MagicMock()
        exa_cls = MagicMock()
        with patch(
            "deep_research_agent.cli.SEARCH_PROVIDERS",
            {
                "tavily": tavily_cls,
                "exa": exa_cls,
            },
        ):
            build_runner(max_iterations=2, search_provider="exa")
        exa_cls.assert_called_once_with()
        tavily_cls.assert_not_called()

    @patch("deep_research_agent.cli.OpenRouterLLMProvider")
    @patch("deep_research_agent.cli.SearchService")
    def test_unknown_provider_raises(self, search_service: MagicMock, llm: MagicMock):
        with self.assertRaises(ValueError):
            build_runner(max_iterations=2, search_provider="nope")


class TestMain(unittest.TestCase):
    def _patch_console(self):
        return patch("builtins.print")

    @patch("builtins.input", return_value="prompt from input")
    @patch("deep_research_agent.cli._run", new_callable=AsyncMock)
    def test_headless_uses_interactive_prompt(self, run, _input):
        run.return_value = "# Report"
        with (
            self._patch_console() as print_mock,
            patch("deep_research_agent.cli.load_environment"),
            patch("deep_research_agent.cli.ResearchAgentStorage") as storage_cls,
        ):
            from deep_research_agent.cli import main

            main([])

        run.assert_awaited_once()
        args, kwargs = run.await_args
        self.assertEqual(args[0], "prompt from input")
        self.assertEqual(args[1], 5)
        self.assertIs(kwargs["storage"], storage_cls.return_value)
        storage_cls.return_value.close.assert_called_once()
        print_mock.assert_called_once_with("# Report")

    @patch("deep_research_agent.tui.app.run_tui")
    @patch("deep_research_agent.cli._run", new_callable=AsyncMock)
    def test_tui_flag_dispatches_to_tui(self, run, run_tui):
        with (
            patch("deep_research_agent.cli.load_environment"),
            patch("deep_research_agent.cli.ResearchAgentStorage") as storage_cls,
        ):
            from deep_research_agent.cli import main

            main(["Research x", "--tui", "--max-iterations", "2", "--model", "m"])
        run.assert_not_awaited()
        run_tui.assert_called_once_with(
            prompt="Research x",
            max_iterations=2,
            model="m",
            search_provider="tavily",
            storage=storage_cls.return_value,
        )
        storage_cls.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
