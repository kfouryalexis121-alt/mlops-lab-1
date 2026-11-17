"""Check model alias resolution at process startup without a tracking server."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.food11 import serve


class ModelStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_alias_change_requires_new_startup(self):
        client = Mock()
        client.get_model_version_by_alias.side_effect = [
            SimpleNamespace(version="1"), SimpleNamespace(version="2")
        ]
        first_model, second_model = object(), object()
        with patch.object(serve.mlflow, "MlflowClient", return_value=client), patch.object(
            serve.mlflow, "set_tracking_uri"
        ), patch.object(
            serve.mlflow.pyfunc, "load_model", side_effect=[first_model, second_model]
        ) as load:
            async with serve.lifespan(serve.app):
                self.assertIs(serve.app.state.model, first_model)
                self.assertEqual(serve.app.state.model_version, "1")
                load.assert_called_once_with("models:/food11/1")
            async with serve.lifespan(serve.app):
                self.assertIs(serve.app.state.model, second_model)
                self.assertEqual(serve.app.state.model_version, "2")
                load.assert_called_with("models:/food11/2")

    async def test_missing_model_fails_startup(self):
        client = Mock()
        client.get_model_version_by_alias.side_effect = RuntimeError("Missing champion")
        with patch.object(serve.mlflow, "MlflowClient", return_value=client), patch.object(
            serve.mlflow, "set_tracking_uri"
        ), patch.object(serve.mlflow.pyfunc, "load_model") as load:
            with self.assertRaisesRegex(RuntimeError, "Missing champion"):
                async with serve.lifespan(serve.app):
                    self.fail("Startup must fail without a registered model")
            load.assert_not_called()
