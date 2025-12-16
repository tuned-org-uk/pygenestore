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
    builder = genestore.store_array(f"./lance_data/{uuid.uuid4().hex}")
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


@pytest.mark.asyncio
async def test_basic_double_store():
    import genestore

    # Create a storage builder and configure it
    builder = genestore.store_array(f"./lance_data/{uuid.uuid4().hex}")
    builder.with_max_rows_per_file(500000)
    builder.with_compression("zstd")

    # Build the storage instance
    storage = builder.build()

    # Create a numpy array (dense matrix)
    np.random.seed(42)  # For reproducibility
    data1 = np.random.randn(1000, 128).astype(np.float64)
    data2 = np.random.randn(1000, 128).astype(np.float64)

    # Store the array (await the async call)
    path1 = await storage.store(data1, "my_dataset_1")
    path2 = await storage.store(data2, "my_dataset_2")

    # Load the array back using the NAME (not path)
    loaded_data1 = await storage.load("my_dataset_1")
    loaded_data2 = await storage.load("my_dataset_2")
    print(f"Loaded shape: {loaded_data1.shape}")

    assert path1 == path2
    assert loaded_data1.shape == loaded_data2.shape == (1000, 128)


def test_builder_creation_and_repr(out_dir):
    builder = genestore.store_array(out_dir)
    r = repr(builder)
    assert "StorageBuilder" in r
    assert out_dir in r


def test_builder_configuration_and_build(out_dir):
    builder = genestore.store_array(out_dir)
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
    storage = genestore.store_array(out_dir).build()

    x = np.random.randn(64, 32).astype(np.float64)
    name = "roundtrip"

    path = await storage.store(x, name)
    assert isinstance(path, str)

    y = await storage.load(name)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.allclose(x, y)


@pytest.mark.asyncio
async def test_store_rejects_empty_array(out_dir):
    storage = genestore.store_array(out_dir).build()

    x = np.zeros((0, 0), dtype=np.float64)

    with pytest.raises(Exception) as e:
        await storage.store(x, "empty")

    msg = str(e.value).lower()
    assert ("empty" in msg) or ("non-zero" in msg) or ("non zero" in msg)


@pytest.mark.asyncio
async def test_store_rejects_non_finite(out_dir):
    storage = genestore.store_array(out_dir).build()

    x = np.random.randn(10, 10).astype(np.float64)
    x[0, 0] = np.nan
    x[1, 1] = np.inf

    with pytest.raises(Exception) as e:
        await storage.store(x, "bad")

    msg = str(e.value)
    assert ("non-finite" in msg) or ("NaN" in msg) or ("Inf" in msg) or ("infinite" in msg.lower())


@pytest.mark.asyncio
async def test_multiple_instances_isolated(out_dir):
    # Two independent storage instances writing into same base but different names.
    # This should work if your backend partitions by name_id internally.
    s1 = genestore.store_array(out_dir).build()

    x1 = np.random.randn(8, 4).astype(np.float64)
    x2 = np.random.randn(8, 4).astype(np.float64)

    print(f"{x1} \n {x2}")

    p1 = await s1.store(x1, "m1")
    p2 = await s1.store(x2, "m2")

    assert p1 == p2
    assert os.path.exists(p1)

    y1 = await s1.load("m1")
    y2 = await s1.load("m2")

    print(f"{y1} \n {y2}")

    assert np.array_equal(x1, y1)
    assert np.array_equal(x2, y2)
