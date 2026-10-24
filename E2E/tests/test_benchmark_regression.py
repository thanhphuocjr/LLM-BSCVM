import unittest
from pathlib import Path

from E2E.tools.benchmark import BENCHMARK_CONTRACTS, evaluate_contract


class TestBenchmarkRegression(unittest.TestCase):
    def test_coinage_benchmark(self):
        res = evaluate_contract("coinage", BENCHMARK_CONTRACTS["coinage"])
        self.assertEqual(res["status"], "PASS", f"Coinage failures: {res.get('failures')}")
        self.assertEqual(res["fused_critical_count"], 0)
        self.assertIn("0.4.18", res["compiler"])

    def test_true_token_benchmark(self):
        res = evaluate_contract("true_token", BENCHMARK_CONTRACTS["true_token"])
        self.assertEqual(res["status"], "PASS", f"TRUEToken failures: {res.get('failures')}")
        self.assertEqual(res["fused_critical_count"], 0)
        self.assertLessEqual(res["fused_count"], 5)
        self.assertIn("0.4.11", res["compiler"])

    def test_elixir_benchmark(self):
        res = evaluate_contract("elixir", BENCHMARK_CONTRACTS["elixir"])
        self.assertEqual(res["status"], "PASS", f"elixir failures: {res.get('failures')}")
        self.assertEqual(res["fused_swc_115_count"], 4)
        self.assertEqual(res["fused_critical_count"], 4)
        self.assertIn("0.4.11", res["compiler"])
        self.assertIn("elixor.returnAmountOfELIXAddressCanProduce", res["empty_body_units"])


if __name__ == "__main__":
    unittest.main()
