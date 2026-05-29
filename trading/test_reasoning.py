"""
Tests for summarize_trade_reason() — mocks the Anthropic client so no API key needed.
Run: python trading/test_reasoning.py
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ── helpers to build a fake anthropic response ────────────────────────────────

def _make_response(text: str):
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# ── tests ─────────────────────────────────────────────────────────────────────

class TestSummarizeTradeReason(unittest.TestCase):

    def _call(self, mock_create, **kwargs):
        """Import fresh each time so patching takes effect."""
        from trading.trader import summarize_trade_reason
        defaults = dict(
            symbol="AAPL",
            price=182.50,
            rsi=23.1,
            sentiment=0.45,
            ma50=178.00,
            headlines=["Apple hits record iPhone sales", "Fed signals rate pause"],
        )
        defaults.update(kwargs)
        return summarize_trade_reason(**defaults)

    @patch("trading.trader.anthropic.Anthropic")
    def test_returns_claude_text(self, MockClient):
        expected = "AAPL is oversold at RSI 23 and trending above its 50-day MA."
        MockClient.return_value.messages.create.return_value = _make_response(expected)

        result = self._call(MockClient.return_value.messages.create)
        self.assertEqual(result, expected)

    @patch("trading.trader.anthropic.Anthropic")
    def test_prompt_contains_key_data(self, MockClient):
        MockClient.return_value.messages.create.return_value = _make_response("ok")
        from trading.trader import summarize_trade_reason

        summarize_trade_reason(
            symbol="NVDA", price=495.00, rsi=22.5,
            sentiment=0.51, ma50=480.00,
            headlines=["NVDA beats earnings"],
        )

        call_args = MockClient.return_value.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]

        self.assertIn("NVDA", prompt)
        self.assertIn("495.00", prompt)
        self.assertIn("22.5", prompt)
        self.assertIn("0.510", prompt)
        self.assertIn("480.00", prompt)
        self.assertIn("NVDA beats earnings", prompt)
        self.assertIn("above", prompt)  # price > ma50

    @patch("trading.trader.anthropic.Anthropic")
    def test_below_ma50_label(self, MockClient):
        MockClient.return_value.messages.create.return_value = _make_response("ok")
        from trading.trader import summarize_trade_reason

        summarize_trade_reason(
            symbol="MSFT", price=300.00, rsi=24.0,
            sentiment=0.35, ma50=320.00,
            headlines=[],
        )

        prompt = MockClient.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("below", prompt)

    @patch("trading.trader.anthropic.Anthropic")
    def test_empty_headlines_fallback(self, MockClient):
        MockClient.return_value.messages.create.return_value = _make_response("ok")
        from trading.trader import summarize_trade_reason

        summarize_trade_reason(
            symbol="GOOGL", price=140.0, rsi=24.9,
            sentiment=0.32, ma50=135.0, headlines=[],
        )

        prompt = MockClient.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("no recent headlines available", prompt)

    @patch("trading.trader.anthropic.Anthropic")
    def test_api_error_returns_empty_string(self, MockClient):
        MockClient.return_value.messages.create.side_effect = Exception("connection refused")
        from trading.trader import summarize_trade_reason

        result = summarize_trade_reason(
            symbol="META", price=350.0, rsi=23.0,
            sentiment=0.40, ma50=340.0, headlines=["Meta AI launch"],
        )
        self.assertEqual(result, "")

    @patch("trading.trader.anthropic.Anthropic")
    def test_strips_whitespace(self, MockClient):
        MockClient.return_value.messages.create.return_value = _make_response("  Good entry.\n")
        from trading.trader import summarize_trade_reason

        result = summarize_trade_reason(
            symbol="AAPL", price=182.0, rsi=24.0,
            sentiment=0.35, ma50=178.0, headlines=[],
        )
        self.assertEqual(result, "Good entry.")


if __name__ == "__main__":
    # run from repo root: python trading/test_reasoning.py
    sys.path.insert(0, ".")
    unittest.main(verbosity=2)
