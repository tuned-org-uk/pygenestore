# pygenestore

Store your `numpy` arrays at scale using the Lance format. Handles millions of rows as far as the memory goes.

## Usage

It is possible to create multiple storages by passing different directories to `create_storage`.

It is possible to store different arrays in the same storage, just set different names.

### Default API (blocking)
```python
import numpy as np
import genestore

# Configure storage
builder = genestore.store_array("./lance_data")
builder.with_max_rows_per_file(500_000)
builder.with_compression("zstd")

# Build storage instance
storage = builder.build()

# Create data (2D float64 numpy array)
np.random.seed(42)
x = np.random.randn(1000, 128).astype(np.float64)

# Store (blocking)
path = storage.store(x, "my_dataset")
print("Stored at:", path)

# Load (blocking)
y = storage.load("my_dataset")
print("Loaded shape:", y.shape)

# Verify roundtrip
assert np.allclose(x, y)
assert np.array_equal(x, y)
print("✓ Data verification passed")
```


### Async API
```python
import numpy as np
import genestore
import asyncio

async def main():
    # Create a storage builder and configure it
    builder = genestore.create_storage(f"./lance_data")
    builder.with_max_rows_per_file(500000)
    builder.with_compression("zstd")

    # Build the storage instance
    storage = builder.build()

    # Create a numpy array (dense matrix)
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(1000, 128).astype(np.float64)

    # Store the array (await the async call)
    path = await storage.aio.store(data, "my_dataset")
    print(f"Storage at: {path}")

    # Load the array back using the NAME (not path)
    loaded_data = await storage.aio.load("my_dataset")
    print(f"Loaded shape: {loaded_data.shape}")

    # Verify the data
    assert np.allclose(data, loaded_data)
    print("✓ Data verification passed!")

if __name__ == "__main__":
    asyncio.run(main())

```

## Tests


```bash
pip install -r requirements-dev.txt
pytest tests/
```
