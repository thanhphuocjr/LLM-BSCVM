import unittest

from E2E.evaluation.benchmark import BENCHMARK_CONTRACTS, evaluate_contract


# Bọc evaluation/benchmark.py thành unit test: 3 contract SmartBugs tham chiếu không được tái phát báo nhầm.
class TestBenchmarkRegression(unittest.TestCase):
    # Coinage: PASS, 0 finding Critical giả, chọn đúng solc 0.4.18.
    def test_coinage_benchmark(self):
        res = evaluate_contract("coinage", BENCHMARK_CONTRACTS["coinage"])
        self.assertEqual(res["status"], "PASS", f"Coinage failures: {res.get('failures')}")
        self.assertEqual(res["fused_critical_count"], 0)
        self.assertIn("0.4.18", res["compiler"])

    # TRUEToken: PASS, 0 Critical (không báo nhầm kill()), <= 5 finding vào agent, solc 0.4.11.
    def test_true_token_benchmark(self):
        res = evaluate_contract("true_token", BENCHMARK_CONTRACTS["true_token"])
        self.assertEqual(res["status"], "PASS", f"TRUEToken failures: {res.get('failures')}")
        self.assertEqual(res["fused_critical_count"], 0)
        self.assertLessEqual(res["fused_count"], 5)
        self.assertIn("0.4.11", res["compiler"])

    # elixir: PASS theo kỳ vọng trong BENCHMARK_CONTRACTS.
    def test_elixir_benchmark(self):
        res = evaluate_contract("elixir", BENCHMARK_CONTRACTS["elixir"])
        self.assertEqual(res["status"], "PASS", f"elixir failures: {res.get('failures')}")
        self.assertEqual(res["fused_swc_115_count"], 4)
        self.assertEqual(res["fused_critical_count"], 4)
        self.assertIn("0.4.11", res["compiler"])
        self.assertIn("elixor.returnAmountOfELIXAddressCanProduce", res["empty_body_units"])


if __name__ == "__main__":
    unittest.main()
