# tests/test_api.py

import os
import uuid
import numpy as np
import pytest

import genestore



@pytest.fixture
def out_dir(tmp_path):
    # Separate dir per test invocation
    d = tmp_path / f"genestore_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

@pytest.mark.asyncio
async def test_basic():
    import genestore

    # Create a storage builder and configure it
    builder = genestore.create_storage(f"./lance_data/{uuid.uuid4().hex}")
    builder.with_max_rows_per_file(500000)
    builder.with_compression("zstd")

    # Build the storage instance
    storage = builder.build()

    # Create a numpy array (dense matrix)
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(1000, 128).astype(np.float64)

    # Store the array (await the async call)
    path = await storage.store(data, "my_dataset")
    print(f"Stored at: {path}")

    # Load the array back using the NAME (not path)
    loaded_data = await storage.load("my_dataset")
    print(f"Loaded shape: {loaded_data.shape}")

    assert path
    assert loaded_data.shape == (1000, 128)

    # Check that 5 random elements match exactly
    np.random.seed(123)  # Different seed for sampling
    for i in range(5):
        row = np.random.randint(0, 1000)
        col = np.random.randint(0, 128)
        original_val = data[row, col]
        loaded_val = loaded_data[row, col]
        
        print(f"Check {i+1}/5: [{row:4d}, {col:3d}] original={original_val:.10f}, loaded={loaded_val:.10f}")
        
        # Assert exact equality (should be bit-identical for Lance format)
        assert original_val == loaded_val, \
            f"Mismatch at [{row}, {col}]: {original_val} != {loaded_val}"
    
    # Also verify overall array equality
    assert np.allclose(data, loaded_data), "Arrays are not close overall"
    assert np.array_equal(data, loaded_data), "Arrays are not exactly equal"
    
    print("✓ All random element checks passed!")


def test_builder_creation_and_repr(out_dir):
    builder = genestore.create_storage(out_dir)
    r = repr(builder)
    assert "StorageBuilder" in r
    assert out_dir in r


def test_builder_configuration_and_build(out_dir):
    builder = genestore.create_storage(out_dir)
    builder.with_max_rows_per_file(500_000)
    builder.with_max_rows_per_group(5_000)
    builder.with_compression("zstd")

    storage = builder.build()
    cfg = storage.get_config()

    assert "max_rows_per_file=500000" in cfg
    assert "max_rows_per_group=5000" in cfg
    assert "compression='zstd'" in cfg

    assert storage.get_output_dir() == out_dir


@pytest.mark.asyncio
async def test_store_and_load_roundtrip(out_dir):
    storage = genestore.create_storage(out_dir).build()

    x = np.random.randn(64, 32).astype(np.float64)
    name = "roundtrip"

    path = await storage.store(x, name)
    assert isinstance(path, str)

    y = await storage.load(name if LOAD_BY_KEY else path)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.allclose(x, y)


@pytest.mark.asyncio
async def test_store_rejects_empty_array(out_dir):
    storage = genestore.create_storage(out_dir).build()

    x = np.zeros((0, 0), dtype=np.float64)

    with pytest.raises(Exception) as e:
        await storage.store(x, "empty")

    msg = str(e.value).lower()
    assert ("empty" in msg) or ("non-zero" in msg) or ("non zero" in msg)


@pytest.mark.asyncio
async def test_store_rejects_non_finite(out_dir):
    storage = genestore.create_storage(out_dir).build()

    x = np.random.randn(10, 10).astype(np.float64)
    x[0, 0] = np.nan
    x[1, 1] = np.inf

    with pytest.raises(Exception) as e:
        await storage.store(x, "bad")

    msg = str(e.value)
    assert ("non-finite" in msg) or ("NaN" in msg) or ("Inf" in msg) or ("infinite" in msg.lower())


@pytest.mark.asyncio
async def test_store_batch_roundtrip(out_dir):
    storage = genestore.create_storage(out_dir).build()

    xs = [np.random.randn(20, 8).astype(np.float64) for _ in range(3)]
    names = ["b1", "b2", "b3"]

    paths = await storage.store_batch(xs, names)
    assert isinstance(paths, list)
    assert len(paths) == 3
    assert all(isinstance(p, str) for p in paths)
    assert all(os.path.exists(p) for p in paths)

    # Load back and compare
    for x, name, path in zip(xs, names, paths):
        y = await storage.load(name if LOAD_BY_KEY else path)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert np.allclose(x, y)


@pytest.mark.asyncio
async def test_store_batch_length_mismatch(out_dir):
    storage = genestore.create_storage(out_dir).build()

    xs = [np.random.randn(5, 3).astype(np.float64) for _ in range(2)]
    names = ["only_one_name"]

    with pytest.raises(Exception) as e:
        await storage.store_batch(xs, names)

    assert "match" in str(e.value).lower()


@pytest.mark.asyncio
async def test_multiple_instances_isolated(out_dir):
    # Two independent storage instances writing into same base but different names.
    # This should work if your backend partitions by name_id internally.
    s1 = genestore.create_storage(out_dir).build()
    s2 = genestore.create_storage(out_dir).build()

    x1 = np.random.randn(8, 4).astype(np.float64)
    x2 = np.random.randn(8, 4).astype(np.float64)

    p1 = await s1.store(x1, "m1")
    p2 = await s2.store(x2, "m2")

    assert os.path.exists(p1)
    assert os.path.exists(p2)

    y1 = await s1.load("m1")
    y2 = await s2.load("m2")

    assert np.allclose(x1, y1)
    assert np.allclose(x2, y2)
