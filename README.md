# pygenestore

Store your `numpy` arrays at scale using the Lance format.

## Usage

```python
import numpy as np
import genestore

# Create a storage builder and configure it
builder = genestore.store_array("./lance_data")
builder.with_max_rows_per_file(500000)
builder.with_compression("zstd")

# Build the storage instance
storage = builder.build()

# Create a numpy array (dense matrix)
data = np.random.randn(1000, 128).astype(np.float64)

# Store the array
path = storage.store(data, "my_dataset")
print(f"Stored at: {path}")

# Load the array back
loaded_data = storage.load(path)
print(f"Loaded shape: {loaded_data.shape}")
```

## Tests


```bash
pip install -r requirements-dev.txt
pytest tests/
```
