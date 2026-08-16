"""runner.py dataset cache.

The grid runner caches parsed dataset instances keyed by dataset name so the
same dataset is created once per group instead of once per prompt. Samples are
read-only, so sharing the instance must not change anything observable.
"""


def test_dataset_cache_creates_once_and_shares_instance(monkeypatch):
    import cotlab.runner as runner_mod

    created = []
    ds_instances = []

    class DummyDataset:
        name = "dummy"

        def __init__(self):
            ds_instances.append(self)

        def sample(self, n, seed=42):
            import random

            rng = random.Random(seed)
            return [f"s{i}" for i in rng.sample(range(10), n)]

    def fake_create(cfg, **kw):
        created.append(cfg)
        return DummyDataset()

    monkeypatch.setattr(runner_mod, "create_component", fake_create)

    cache = {}
    first = runner_mod.get_cached_dataset(cache, "dummy", "DummyDataset")
    second = runner_mod.get_cached_dataset(cache, "dummy", "DummyDataset")

    assert first is second, "cache must return the same instance on repeat lookup"
    assert len(created) == 1, "dataset must be created exactly once (cache hit)"
    assert len(ds_instances) == 1, "one shared instance across jobs"
    assert first.sample(4) == first.sample(4), "sampling must stay deterministic"


def test_dataset_cache_keyed_by_name(monkeypatch):
    import cotlab.runner as runner_mod

    created = []

    def fake_create(cfg, **kw):
        created.append(cfg)
        return type("DS", (), {"name": cfg})()

    monkeypatch.setattr(runner_mod, "create_component", fake_create)

    cache = {}
    a = runner_mod.get_cached_dataset(cache, "ds_a", "A")
    b = runner_mod.get_cached_dataset(cache, "ds_b", "B")
    a2 = runner_mod.get_cached_dataset(cache, "ds_a", "A")

    assert a is a2, "same name must hit the cache"
    assert b is not a, "different names must create different instances"
    assert created == ["A", "B"], "one creation per distinct name"
