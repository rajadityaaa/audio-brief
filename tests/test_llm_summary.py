import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_brief.llm import summarize_with_ollama
from audio_brief.writers import write_summary


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class TestLLMSummary(unittest.TestCase):
    @patch("audio_brief.llm.request.urlopen")
    def test_ollama_summary(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(
            {"response": "The team agreed to release the project next week."}
        )

        result = summarize_with_ollama(
            "The team discussed the project and agreed to release it next week.",
            model="test-model",
            url="http://localhost:11434/api/generate",
        )

        self.assertEqual(
            result,
            "The team agreed to release the project next week.",
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(payload["model"], "test-model")
        self.assertFalse(payload["stream"])
        self.assertIn("The team discussed the project", payload["prompt"])

    def test_extractive_summary_is_default(self):
        text = (
            "The team discussed the new project. "
            "They decided to release the first version next week."
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            result = write_summary(text, path)

            self.assertEqual(
                result,
                [
                    "The team discussed the new project.",
                    "They decided to release the first version next week.",
                ],
            )
            self.assertIn("1. The team discussed the new project.", path.read_text())

    @patch("audio_brief.llm.summarize_with_ollama")
    def test_llm_failure_falls_back_to_extractive(self, mock_summary):
        mock_summary.side_effect = RuntimeError("Ollama unavailable")

        text = (
            "The team discussed the new project. "
            "They decided to release the first version next week."
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            result = write_summary(text, path, llm=True)

            self.assertTrue(result)
            self.assertIn("1. The team discussed the new project.", path.read_text())


if __name__ == "__main__":
    unittest.main()
