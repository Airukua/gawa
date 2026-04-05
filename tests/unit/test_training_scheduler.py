import torch

from training.scheduler import SchedulerConfig, build_scheduler


def test_scheduler_constant_returns_none():
    model = torch.nn.Linear(2, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = SchedulerConfig(name="constant")
    sched = build_scheduler(optim, total_steps=10, config=cfg)
    assert sched is None


def test_scheduler_cosine_returns_scheduler():
    model = torch.nn.Linear(2, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = SchedulerConfig(name="cosine", warmup_steps=0, min_lr=0.0)
    sched = build_scheduler(optim, total_steps=10, config=cfg)
    assert sched is not None
