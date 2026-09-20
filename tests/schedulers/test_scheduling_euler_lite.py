"""TEMPLATE — copy to tests/schedulers/test_scheduling_<snake>.py and set the
two constants below. Runs with zero installs (structural + signature contract)
and adds behavioral checks when torch + diffusers are available.

See examples/scaffolded_scheduler + tests/schedulers/test_scheduling_ddpm_lite.py
for a filled-in example.
"""
import ast
import importlib.util
import unittest
from pathlib import Path

# ---- edit these two for your scheduler -------------------------------------
TARGET = Path(__file__).resolve().parents[2] / "src" / "diffusers" / "schedulers" / "scheduling_euler_lite.py"
CLASS = "EulerLiteScheduler"
# ----------------------------------------------------------------------------


def _class_node():
    for node in ast.walk(ast.parse(TARGET.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == CLASS:
            return node
    raise AssertionError(f"{CLASS} not found in {TARGET}")


def _method(node, name):
    for n in node.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


class TestStructuralContract(unittest.TestCase):
    """Thin structural sanity — the convention gate is the primary owner."""

    @classmethod
    def setUpClass(cls):
        cls.node = _class_node()

    def test_inherits_required_mixins(self):
        bases = {getattr(b, "id", getattr(b, "attr", "")) for b in self.node.bases}
        self.assertEqual({"SchedulerMixin", "ConfigMixin"} & bases, {"SchedulerMixin", "ConfigMixin"})


class TestSignatureContract(unittest.TestCase):
    """Deeper than the gate: verify the SIGNATURES, not just method presence."""

    @classmethod
    def setUpClass(cls):
        cls.node = _class_node()

    def test_set_timesteps_signature(self):
        m = _method(self.node, "set_timesteps")
        self.assertIsNotNone(m, "set_timesteps missing")
        args = [a.arg for a in m.args.args]
        self.assertIn("num_inference_steps", args,
                      "set_timesteps must accept num_inference_steps (verified against upstream source)")
        self.assertIn("device", args, "set_timesteps must accept device")

    def test_step_signature(self):
        m = _method(self.node, "step")
        self.assertIsNotNone(m, "step missing")
        args = [a.arg for a in m.args.args]
        for expected in ("model_output", "timestep", "sample", "generator"):
            self.assertIn(expected, args, f"step must accept {expected} (upstream contract)")


class TestBehavioralContract(unittest.TestCase):
    """Real behaviour — runs in CI with torch; skipped cleanly offline."""

    def setUp(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed; behavioral contract runs in CI")
        if importlib.util.find_spec("diffusers") is None:
            self.skipTest("diffusers not installed; behavioral contract runs in CI")
        spec = importlib.util.spec_from_file_location(CLASS, TARGET)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.Scheduler = getattr(self.mod, CLASS)

    def test_config_roundtrip(self):
        s = self.Scheduler(num_train_timesteps=500)
        rebuilt = self.Scheduler.from_config(s.config)
        self.assertEqual(rebuilt.config.num_train_timesteps, 500)

    def test_timesteps_count(self):
        s = self.Scheduler()
        s.set_timesteps(25)
        self.assertEqual(len(s.timesteps), 25)

    def test_output_type(self):
        import torch
        s = self.Scheduler()
        s.set_timesteps(10)
        sample = torch.zeros(1, 3, 8, 8)
        out = s.step(torch.ones_like(sample), 1, sample, generator=torch.Generator().manual_seed(0))
        self.assertTrue(hasattr(out, "prev_sample"))
        self.assertEqual(out.prev_sample.shape, sample.shape)
        self.assertEqual(out.prev_sample.dtype, sample.dtype)

    def test_same_seed_same_output(self):
        import torch
        s = self.Scheduler()
        s.set_timesteps(10)
        sample = torch.zeros(1, 3, 8, 8)
        mo = torch.ones_like(sample)
        a = s.step(mo, 1, sample, generator=torch.Generator().manual_seed(0)).prev_sample
        b = s.step(mo, 1, sample, generator=torch.Generator().manual_seed(0)).prev_sample
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
