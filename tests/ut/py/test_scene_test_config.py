# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for SceneTestCase runtime config construction."""

from __future__ import annotations

from simpler_setup.scene_test import SceneTestCase


class _ConfigScene(SceneTestCase):
    pass


def test_build_config_uses_case_aicpu_thread_num(monkeypatch):
    monkeypatch.delenv("SIMPLER_AICPU_THREAD_NUM", raising=False)
    monkeypatch.delenv("SIMPLER_BLOCK_DIM", raising=False)

    config = _ConfigScene()._build_config({"aicpu_thread_num": 4, "block_dim": 24})

    assert config.aicpu_thread_num == 4
    assert config.block_dim == 24


def test_build_config_env_overrides_aicpu_thread_num(monkeypatch):
    monkeypatch.setenv("SIMPLER_AICPU_THREAD_NUM", "6")

    config = _ConfigScene()._build_config({"aicpu_thread_num": 4, "block_dim": 24})

    assert config.aicpu_thread_num == 6


def test_build_config_invalid_env_keeps_case_aicpu_thread_num(monkeypatch):
    monkeypatch.setenv("SIMPLER_AICPU_THREAD_NUM", "invalid")

    config = _ConfigScene()._build_config({"aicpu_thread_num": 4, "block_dim": 24})

    assert config.aicpu_thread_num == 4


def test_build_config_env_overrides_block_dim(monkeypatch):
    monkeypatch.setenv("SIMPLER_BLOCK_DIM", "20")

    config = _ConfigScene()._build_config({"aicpu_thread_num": 4, "block_dim": 24})

    assert config.block_dim == 20


def test_build_config_invalid_env_keeps_case_block_dim(monkeypatch):
    monkeypatch.setenv("SIMPLER_BLOCK_DIM", "invalid")

    config = _ConfigScene()._build_config({"aicpu_thread_num": 4, "block_dim": 24})

    assert config.block_dim == 24


def test_build_config_uses_case_pipeline_strategy():
    config = _ConfigScene()._build_config({"pipeline_strategy": 2})

    assert config.pipeline_strategy == 2


def test_build_config_invalid_pipeline_strategy_uses_default():
    config = _ConfigScene()._build_config({"pipeline_strategy": "invalid"})

    assert config.pipeline_strategy == -1
