import unittest

from E2E.common.solc import select_best_solc
from E2E.config import SMARTBUGS_WILD_DIR
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1_fusion.fusion import fuse_findings
from E2E.phase1a_static.static_runner import run_static


# 3 contract SmartBugs-Wild tham chiếu: chọn solc, tách hàm, static + fusion không được tái phát báo nhầm.
BENCHMARK_FILES = {
    "coinage": "0x7415c7bf3e2415fa9a55f1fd8b6fccf2914c39a6.sol",
    "true_token": "0xa4d17ab1ee0efdd23edc2869e7ba96b89eecf9ab.sol",
    "elixir": "0xc8c6a31a4a806d3710a7b38b7b296d2fabccdba8.sol",
}


# Chạy Phase 0 -> 1A -> fusion (không detector) trên một contract tham chiếu.
def evaluate_contract(key: str) -> dict:
    path = SMARTBUGS_WILD_DIR / BENCHMARK_FILES[key]
    if not path.exists():
        raise unittest.SkipTest(f"benchmark contract not found: {path}")
    code = path.read_text(encoding="utf-8")
    solc_bin, _, _ = select_best_solc(code)
    units = parse_code_units(code)
    static_res = run_static(code, units=[unit for unit in units if unit.has_body])
    fused, _ = fuse_findings([], static_res, {"findings": []})
    return {
        "compiler": solc_bin.name if solc_bin else "none",
        "empty_body_units": [unit.unit_id for unit in units if not unit.has_body],
        "fused_count": len(fused),
        "fused_critical_count": sum(1 for f in fused if f.get("severity") == "Critical"),
        "fused_swc_115_count": sum(1 for f in fused if f.get("swc") == "SWC-115"),
    }


# Hồi quy static + fusion trên 3 contract tham chiếu.
class TestBenchmarkRegression(unittest.TestCase):
    # Coinage: 0 finding Critical giả, <= 5 finding vào agent, chọn đúng solc 0.4.18.
    def test_coinage_benchmark(self):
        res = evaluate_contract("coinage")
        self.assertEqual(res["fused_critical_count"], 0)
        self.assertLessEqual(res["fused_count"], 5)
        self.assertIn("0.4.18", res["compiler"])

    # TRUEToken: 0 Critical (không báo nhầm kill()), <= 5 finding vào agent, solc 0.4.11.
    def test_true_token_benchmark(self):
        res = evaluate_contract("true_token")
        self.assertEqual(res["fused_critical_count"], 0)
        self.assertLessEqual(res["fused_count"], 5)
        self.assertIn("0.4.11", res["compiler"])

    # elixir: đúng 4 finding SWC-115 (tx.origin), hàm không thân được nhận diện, solc 0.4.11.
    def test_elixir_benchmark(self):
        res = evaluate_contract("elixir")
        self.assertEqual(res["fused_swc_115_count"], 4)
        self.assertEqual(res["fused_critical_count"], 4)
        self.assertIn("0.4.11", res["compiler"])
        self.assertIn("elixor.returnAmountOfELIXAddressCanProduce", res["empty_body_units"])


if __name__ == "__main__":
    unittest.main()
