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


def test_basic_blocking():
    # Create a storage builder and configure it
    builder = genestore.store_array(f"./lance_data/{uuid.uuid4().hex}")
    builder.with_max_rows_per_file(500000)
    builder.with_compression("zstd")

    # Build the storage instance
    storage = builder.build()

    # Create a numpy array (dense matrix)
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(1000, 128).astype(np.float64)

    # Store the array (blocking call)
    path = storage.store(data, "my_dataset")
    print(f"Stored at: {path}")

    # Load the array back using the NAME (not path)
    loaded_data = storage.load("my_dataset")
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
        print(
            f"Check {i+1}/5: [{row:4d}, {col:3d}] "
            f"original={original_val:.10f}, loaded={loaded_val:.10f}"
        )
        assert original_val == loaded_val, (
            f"Mismatch at [{row}, {col}]: {original_val} != {loaded_val}"
        )

    # Also verify overall array equality
    assert np.allclose(data, loaded_data), "Arrays are not close overall"
    assert np.array_equal(data, loaded_data), "Arrays are not exactly equal"
    print("✓ All random element checks passed!")


def test_basic_double_store_blocking():
    builder = genestore.store_array(f"./lance_data/{uuid.uuid4().hex}")
    builder.with_max_rows_per_file(500000)
    builder.with_compression("zstd")
    storage = builder.build()

    np.random.seed(42)
    data1 = np.random.randn(1000, 128).astype(np.float64)
    data2 = np.random.randn(1000, 128).astype(np.float64)

    path1 = storage.store(data1, "my_dataset_1")
    path2 = storage.store(data2, "my_dataset_2")

    loaded_data1 = storage.load("my_dataset_1")
    loaded_data2 = storage.load("my_dataset_2")

    assert path1 == path2
    assert loaded_data1.shape == loaded_data2.shape == (1000, 128)


def test_builder_creation_and_repr_blocking(out_dir):
    builder = genestore.store_array(out_dir)
    r = repr(builder)
    assert "StorageBuilder" in r
    assert out_dir in r


def test_builder_configuration_and_build_blocking(out_dir):
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


def test_store_and_load_roundtrip_blocking(out_dir):
    storage = genestore.store_array(out_dir).build()

    x = np.random.randn(64, 32).astype(np.float64)
    name = "roundtrip"

    path = storage.store(x, name)
    assert isinstance(path, str)

    y = storage.load(name)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.allclose(x, y)


def test_store_rejects_empty_array_blocking(out_dir):
    storage = genestore.store_array(out_dir).build()
    x = np.zeros((0, 0), dtype=np.float64)

    with pytest.raises(Exception) as e:
        storage.store(x, "empty")

    msg = str(e.value).lower()
    assert ("empty" in msg) or ("non-zero" in msg) or ("non zero" in msg)


def test_store_rejects_non_finite_blocking(out_dir):
    storage = genestore.store_array(out_dir).build()

    x = np.random.randn(10, 10).astype(np.float64)
    x[0, 0] = np.nan
    x[1, 1] = np.inf

    with pytest.raises(Exception) as e:
        storage.store(x, "bad")

    msg = str(e.value)
    assert ("non-finite" in msg) or ("NaN" in msg) or ("Inf" in msg) or ("infinite" in msg.lower())


def test_multiple_instances_isolated_blocking(out_dir):
    # Two independent storage instances writing into same base but different names.
    s1 = genestore.store_array(out_dir).build()

    x1 = np.random.randn(8, 4).astype(np.float64)
    x2 = np.random.randn(8, 4).astype(np.float64)

    p1 = s1.store(x1, "m1")
    p2 = s1.store(x2, "m2")

    assert p1 == p2
    assert os.path.exists(p1)

    y1 = s1.load("m1")
    y2 = s1.load("m2")

    assert np.array_equal(x1, y1)
    assert np.array_equal(x2, y2)


def test_store_accepts_vstack_arrays_blocking(out_dir):
    """Test that store() can handle arrays created with np.vstack()"""
    storage = genestore.store_array(out_dir).build()

    # Create multiple batches of embeddings (simulating batched processing)
    np.random.seed(42)
    batch1 = np.random.randn(100, 128).astype(np.float64)
    batch2 = np.random.randn(100, 128).astype(np.float64)
    batch3 = np.random.randn(100, 128).astype(np.float64)

    # Stack them vertically using np.vstack (this creates a different array type)
    stacked_embeddings = np.vstack([batch1, batch2, batch3])

    # Verify the stacked array has expected shape
    assert stacked_embeddings.shape == (300, 128)

    # Store the vstack-created array (this should work without casting errors)
    path = storage.store(stacked_embeddings, "vstacked_data")
    assert isinstance(path, str)
    assert os.path.exists(path)

    # Load it back
    loaded_data = storage.load("vstacked_data")
    assert isinstance(loaded_data, np.ndarray)
    assert loaded_data.shape == (300, 128)

    # Verify data integrity
    assert np.allclose(stacked_embeddings, loaded_data)
    assert np.array_equal(stacked_embeddings, loaded_data)

    # Verify individual batches are preserved correctly
    assert np.array_equal(loaded_data[:100, :], batch1)
    assert np.array_equal(loaded_data[100:200, :], batch2)
    assert np.array_equal(loaded_data[200:, :], batch3)

    print(f"✓ Successfully stored and loaded vstack array: {stacked_embeddings.shape}")


def test_store_accepts_various_array_constructions_blocking(out_dir):
    """Test that store() handles arrays created through different NumPy operations"""
    storage = genestore.store_array(out_dir).build()

    np.random.seed(99)
    base_data = np.random.randn(50, 64).astype(np.float64)

    # Test different array construction methods
    test_cases = [
        ("vstack", np.vstack([base_data, base_data])),
        ("hstack", np.hstack([base_data, base_data])),
        ("concatenate", np.concatenate([base_data, base_data], axis=0)),
        ("copy", base_data.copy()),
        ("slice", base_data[10:40, :]),
        ("transpose_twice", base_data.T.T),
    ]

    for name, array in test_cases:
        # Ensure array is contiguous and float64
        array = np.ascontiguousarray(array, dtype=np.float64)

        # Store the array
        path = storage.store(array, f"array_{name}")
        assert isinstance(path, str), f"Failed to store {name} array"

        # Load it back
        loaded = storage.load(f"array_{name}")
        assert loaded.shape == array.shape, f"Shape mismatch for {name}"
        assert np.array_equal(array, loaded), f"Data mismatch for {name}"

        print(f"✓ {name}: {array.shape} stored and loaded successfully")

    print("✓ All array construction methods handled correctly!")
