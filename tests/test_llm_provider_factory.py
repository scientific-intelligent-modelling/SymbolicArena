import os
import unittest
from unittest import mock

from scientific_intelligent_modelling.algorithms.llmsr_wrapper.llmsr import llm as llmsr_llm
from scientific_intelligent_modelling.algorithms.drsr_wrapper.drsr import llm as drsr_llm
from scientific_intelligent_modelling.srkit import llm as srkit_llm


class LLMProviderFactoryTest(unittest.TestCase):
    def test_llmsr_mock_factory_returns_fixed_response_without_network(self):
        client = llmsr_llm.ClientFactory.from_config(
            {
                "model": "mock/fixed",
                "fixed_response": "    return x0 + params[0]\n",
                "mock_prompt_tokens": 7,
                "mock_content_tokens": 3,
            }
        )
        self.assertIsInstance(client, llmsr_llm.MockFixedClient)
        with mock.patch.object(llmsr_llm.requests, "post") as mocked_post:
            response = client.chat([{"role": "user", "content": "hello"}])
        mocked_post.assert_not_called()
        self.assertEqual(response["content"], "    return x0 + params[0]\n")
        self.assertEqual(response["tokens"]["prompt"], 7)
        self.assertEqual(response["tokens"]["content"], 3)

    def test_llmsr_mock_pool_factory_cycles_responses_without_network(self):
        client = llmsr_llm.ClientFactory.from_config(
            {
                "model": "mock/pool",
                "mock_responses": [
                    "    return x0 + params[0]\n",
                    "    return x1 + params[0]\n",
                ],
            }
        )
        self.assertIsInstance(client, llmsr_llm.MockPoolClient)
        with mock.patch.object(llmsr_llm.requests, "post") as mocked_post:
            first = client.chat([{"role": "user", "content": "hello"}])
            second = client.chat([{"role": "user", "content": "hello"}])
            third = client.chat([{"role": "user", "content": "hello"}])
        mocked_post.assert_not_called()
        self.assertEqual(first["content"], "    return x0 + params[0]\n")
        self.assertEqual(second["content"], "    return x1 + params[0]\n")
        self.assertEqual(third["content"], "    return x0 + params[0]\n")

    def test_llmsr_deepinfra_factory_defaults(self):
        with mock.patch.dict(os.environ, {"DEEPINFRA_API_KEY": "dummy-deepinfra-key"}, clear=False):
            client = llmsr_llm.ClientFactory.from_config(
                {"model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct"}
            )
        self.assertIsInstance(client, llmsr_llm.DeepInfraClient)
        self.assertEqual(client.base_url, "https://api.deepinfra.com/v1/openai")
        self.assertEqual(client.model, "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.assertEqual(client.api_key, "dummy-deepinfra-key")

    def test_llmsr_deepinfra_factory_requires_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPINFRA_API_KEY"):
                llmsr_llm.ClientFactory.from_config(
                    {"model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct"}
                )

    def test_drsr_deepinfra_factory_supports_alias(self):
        client = drsr_llm.ClientFactory.from_config(
            {
                "model": "deep-infra/meta-llama/Meta-Llama-3.1-8B-Instruct",
                "api_key": "dummy",
            }
        )
        self.assertIsInstance(client, drsr_llm.DeepInfraClient)
        self.assertEqual(client.base_url, "https://api.deepinfra.com/v1/openai")

    def test_drsr_deepinfra_factory_requires_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPINFRA_API_KEY"):
                drsr_llm.ClientFactory.from_config(
                    {"model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct"}
                )

    def test_srkit_deepinfra_factory_honors_explicit_base_url(self):
        client = srkit_llm.ClientFactory.from_config(
            {
                "model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
                "api_key": "dummy",
                "base_url": "https://api.deepinfra.com/v1/openai",
            }
        )
        self.assertIsInstance(client, srkit_llm.DeepInfraClient)
        self.assertEqual(client.base_url, "https://api.deepinfra.com/v1/openai")

    def test_srkit_cstcloud_factory_supports_benchmark_provider(self):
        with mock.patch.dict(os.environ, {"CSTCLOUD_API_KEY": "dummy-cstcloud-key"}, clear=False):
            client = srkit_llm.ClientFactory.from_config({"model": "CSTCloud/gpt-oss-120b"})
        self.assertIsInstance(client, srkit_llm.CSTCloudClient)
        self.assertEqual(client.base_url, "https://uni-api.cstcloud.cn/v1")
        self.assertEqual(client.model, "gpt-oss-120b")
        self.assertEqual(client.api_key, "dummy-cstcloud-key")


if __name__ == "__main__":
    unittest.main()
