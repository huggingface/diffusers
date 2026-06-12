import unittest

from diffusers.utils.triton_compat import patch_triton_autotuner_prune_configs


class TritonCompatTests(unittest.TestCase):
    def test_patch_materializes_iterator_results(self):
        class FakeAutotuner:
            def prune_configs(self, kwargs):
                return (value for value in kwargs["values"])

        fake_module = type("FakeModule", (), {"Autotuner": FakeAutotuner})

        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"triton.runtime.autotuner": fake_module}):
            patched = patch_triton_autotuner_prune_configs()
            self.assertTrue(patched)
            result = FakeAutotuner().prune_configs({"values": [1, 2, 3]})
            self.assertEqual(result, [1, 2, 3])

            patched_again = patch_triton_autotuner_prune_configs()
            self.assertFalse(patched_again)


if __name__ == "__main__":
    unittest.main()
