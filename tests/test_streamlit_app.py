import unittest
from pathlib import Path


class StreamlitAppTest(unittest.TestCase):
    def test_app_file_exists(self) -> None:
        content = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("Dashboard", content)
        self.assertIn("Prediction", content)
        self.assertIn("Insights", content)


if __name__ == "__main__":
    unittest.main()

