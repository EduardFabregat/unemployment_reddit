import pytest

from unemployment_reddit import cli, pipeline
from unemployment_reddit.config import Config


def test_config_artifact_paths_use_tag(tmp_path):
    cfg = Config(data_dir=tmp_path, tag="t1")
    assert cfg.model_path.name == "unemployment_bertopic_model_t1"
    assert cfg.meta_theta_path.name == "meta_theta_df_t1.parquet"
    assert cfg.db_path.parent == tmp_path


def test_cli_dispatches_stage_with_overrides(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(pipeline, "run_assign", lambda cfg, **kw: calls.update(cfg=cfg, **kw))
    cli.main(
        ["--data-dir", str(tmp_path), "--tag", "x", "assign", "--skip-emotions", "--emotion-batch-size", "64"]
    )
    assert calls["cfg"].tag == "x"
    assert calls["with_emotions"] is False
    assert calls["batch_size"] == 64


def test_cli_requires_a_stage():
    with pytest.raises(SystemExit):
        cli.main([])
