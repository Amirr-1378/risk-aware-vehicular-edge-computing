import unittest

import risk_aware_v2x


class PackageImportTests(unittest.TestCase):
    def test_package_version(self) -> None:
        self.assertEqual(risk_aware_v2x.__version__, "0.1.0")
