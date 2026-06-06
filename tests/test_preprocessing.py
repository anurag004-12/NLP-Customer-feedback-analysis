import unittest

from src.preprocessing.text_cleaner import TextPreprocessor


class TextPreprocessorTest(unittest.TestCase):
    def test_clean_removes_noise(self) -> None:
        cleaner = TextPreprocessor()
        text = "Great!!! Visit https://example.com <b>NOW</b> 😊"
        self.assertEqual(cleaner.clean(text), "great visit now")


if __name__ == "__main__":
    unittest.main()

