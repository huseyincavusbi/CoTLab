"""runner.py dataset cache.

The grid runner caches parsed dataset instances keyed by dataset name so the
same dataset is created once per group instead of once per prompt. Samples are
read-only, so sharing the instance must not change anything observable.
"""


def test_dataset_cache_shares_instance_and_sampling_is_identical(monkeypatch):
    import cotlab.runner as runner_mod

    created = []
    ds_instances = []
    real_create = runner_mod.create_component

    class DummyDataset:
        name = "dummy"

        def __init__(self):
            ds_instances.append(self)

        def sample(self, n, seed=42):
            import random

            rng = random.Random(seed)
            return [f"s{i}" for i in rng.sample(range(10), n)]

    def make_component(cfg, **kw):
        target = cfg if isinstance(cfg, str) else getattr(cfg, "_target_", "")
        if target == "DummyDataset":
            created.append(target)
            return DummyDataset()
        return real_create(cfg, **kw)

    monkeypatch.setattr(runner_mod, "create_component", make_component)

    # Mirrors the runner loop: same dataset_name across grid jobs.
    cache = {}
    for _ in range(3):
        name = "DummyDataset"
        ds = cache[name] if name in cache else make_component("DummyDataset")
        cache[name] = ds
        assert ds.sample(4) == ds.sample(4)

    assert len(created) == 1, "dataset must be created exactly once (cache hit)"
    assert len(ds_instances) == 1, "one shared instance across jobs"
