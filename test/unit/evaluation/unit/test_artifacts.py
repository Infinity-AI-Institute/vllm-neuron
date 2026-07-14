# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from test.evaluation.utils.artifacts import ArtifactManager


@pytest.fixture
def artifact_manager(tmp_path):
    return ArtifactManager(base_dir=tmp_path)


@pytest.fixture
def mock_model_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return str(model_dir)


class TestSetupModelPath:
    def test_step1_existing_path(self, artifact_manager, mock_model_path):
        """Step 1: Use model_path if it exists"""
        model_config = {"model_path": mock_model_path}
        result = artifact_manager.setup_model_path(model_config)
        assert result == mock_model_path

    def test_step2_s3_download(self, artifact_manager, tmp_path):
        """Step 2: Download from S3 if model_s3_path is provided"""
        model_config = {
            "model_path": str(tmp_path / "downloaded_model"),
            "model_s3_path": "s3://bucket/model",
        }

        with patch("test.evaluation.utils.artifacts.download_from_s3") as mock_download:
            result = artifact_manager.setup_model_path(model_config)
            mock_download.assert_called_once_with(
                "s3://bucket/model", model_config["model_path"]
            )
            assert result == model_config["model_path"]

    def test_step3_resolve_model_dir(self, artifact_manager, mock_model_path):
        """Step 3: Use resolve_model_dir for FSx/SSD lookup"""
        model_config = {"model_path": "meta-llama/Llama-3.1-8B"}

        with patch("test.evaluation.utils.artifacts.resolve_model_dir") as mock_resolve:
            mock_resolve.return_value = (mock_model_path, True)
            result = artifact_manager.setup_model_path(model_config)
            mock_resolve.assert_called_once_with("meta-llama/Llama-3.1-8B")
            assert result == mock_model_path

    def test_step3_resolve_fails(self, artifact_manager):
        """Step 3: Return model_path if resolve_model_dir fails"""
        model_config = {"model_path": "some/path"}

        with patch(
            "test.evaluation.utils.artifacts.resolve_model_dir",
            side_effect=Exception("Import error"),
        ):
            result = artifact_manager.setup_model_path(model_config)
            assert result == "some/path"

    def test_s3_takes_precedence_over_resolve(self, artifact_manager, tmp_path):
        """Step 2 (S3) should execute before Step 3 (resolve_model_dir)"""
        model_config = {
            "model_path": "meta-llama/Llama-3.1-8B",
            "model_s3_path": "s3://bucket/model",
        }

        with (
            patch("test.evaluation.utils.artifacts.download_from_s3") as mock_download,
            patch("test.evaluation.utils.artifacts.resolve_model_dir") as mock_resolve,
        ):
            _ = artifact_manager.setup_model_path(model_config)
            mock_download.assert_called_once()
            mock_resolve.assert_not_called()
