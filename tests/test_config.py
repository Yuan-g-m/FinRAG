import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import Config


class ConfigSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_finance_sources(self):
        self.assertEqual(self.config.VALID_SOURCES, ["股票", "基金", "债券", "银行", "保险"])

    def test_finance_database(self):
        self.assertEqual(self.config.MYSQL_DATABASE, "finance_kg")
        self.assertEqual(self.config.MILVUS_DATABASE_NAME, "finance_rag")
        self.assertEqual(self.config.MILVUS_COLLECTION_NAME, "finrag_final")

    def test_llm_base_url(self):
        self.assertTrue(self.config.LLM_BASE_URL.startswith("https://"))


if __name__ == "__main__":
    unittest.main()
