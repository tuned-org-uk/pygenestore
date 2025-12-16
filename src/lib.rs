use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyException;
use std::path::PathBuf;

// Storage builder for configuring Lance storage parameters
#[pyclass]
#[derive(Clone)]
pub struct StorageBuilder {
    output_dir: PathBuf,
    max_rows_per_file: Option<usize>,
    max_rows_per_group: Option<usize>,
    compression: Option<String>,
}

#[pymethods]
impl StorageBuilder {
    #[new]
    #[pyo3(signature = (output_dir, max_rows_per_file=None, max_rows_per_group=None, compression=None))]
    fn new(
        output_dir: String,
        max_rows_per_file: Option<usize>,
        max_rows_per_group: Option<usize>,
        compression: Option<String>,
    ) -> Self {
        Self {
            output_dir: PathBuf::from(output_dir),
            max_rows_per_file,
            max_rows_per_group,
            compression,
        }
    }

    /// Set the output directory for Lance storage
    fn with_output_dir(&mut self, output_dir: String) -> PyResult<()> {
        self.output_dir = PathBuf::from(output_dir);
        Ok(())
    }

    /// Set maximum rows per file
    fn with_max_rows_per_file(&mut self, max_rows: usize) -> PyResult<()> {
        self.max_rows_per_file = Some(max_rows);
        Ok(())
    }

    /// Set maximum rows per group
    fn with_max_rows_per_group(&mut self, max_rows: usize) -> PyResult<()> {
        self.max_rows_per_group = Some(max_rows);
        Ok(())
    }

    /// Set compression algorithm (e.g., "zstd", "lz4", "none")
    fn with_compression(&mut self, compression: String) -> PyResult<()> {
        self.compression = Some(compression);
        Ok(())
    }

    /// Build the storage instance
    fn build(&self) -> PyResult<LanceStorage> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| PyException::new_err(format!("Failed to create directory: {}", e)))?;

        Ok(LanceStorage {
            output_dir: self.output_dir.clone(),
            max_rows_per_file: self.max_rows_per_file.unwrap_or(1_000_000),
            max_rows_per_group: self.max_rows_per_group.unwrap_or(10_000),
            compression: self.compression.clone().unwrap_or_else(|| "zstd".to_string()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageBuilder(output_dir='{}', max_rows_per_file={:?}, max_rows_per_group={:?}, compression={:?})",
            self.output_dir.display(),
            self.max_rows_per_file,
            self.max_rows_per_group,
            self.compression
        )
    }
}

// Main storage interface for Lance operations
#[pyclass]
pub struct LanceStorage {
    output_dir: PathBuf,
    max_rows_per_file: usize,
    max_rows_per_group: usize,
    compression: String,
}

#[pymethods]
impl LanceStorage {
    /// Store a numpy array to Lance format
    /// 
    /// Parameters
    /// ----------
    /// array : numpy.ndarray
    ///     2D numpy array to store (dense matrix)
    /// name : str
    ///     Name of the dataset/file
    /// 
    /// Returns
    /// -------
    /// str
    ///     Path to the stored Lance dataset
    #[pyo3(signature = (array, name))]
    fn store<'py>(
        &self,
        py: Python<'py>,
        array: PyReadonlyArray2<f64>,
        name: String,
    ) -> PyResult<String> {
        // Release GIL during computation
        py.allow_threads(|| {
            // Convert numpy array to ndarray view
            let array_view = array.as_array();
            let shape = array_view.shape();
            let rows = shape[0];
            let cols = shape[1];

            // Construct output path
            let output_path = self.output_dir.join(format!("{}.lance", name));

            // TODO: Use genegraph-storage crate to write to Lance format
            // This is a placeholder for the actual implementation
            // You would call genegraph-storage methods here like:
            // 
            // use genegraph_storage::lance::LanceWriter;
            // let mut writer = LanceWriter::new(&output_path, self.max_rows_per_file)?;
            // writer.write_dense_matrix(array_view)?;
            // writer.close()?;

            // For now, we return a success message
            // Replace this with actual genegraph-storage implementation
            Ok(output_path.to_string_lossy().to_string())
        })
    }

    /// Load a Lance dataset into a numpy array
    /// 
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the Lance dataset file
    /// 
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     2D numpy array containing the loaded data
    #[pyo3(signature = (path))]
    fn load<'py>(
        &self,
        py: Python<'py>,
        path: String,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Release GIL during I/O
        let data = py.allow_threads(|| {
            let lance_path = PathBuf::from(&path);

            // TODO: Use genegraph-storage crate to read from Lance format
            // This is a placeholder for the actual implementation
            // 
            // use genegraph_storage::lance::LanceReader;
            // let reader = LanceReader::open(&lance_path)?;
            // let array = reader.read_dense_matrix()?;
            // Ok(array)

            // Placeholder: return dummy data
            // Replace with actual genegraph-storage implementation
            let rows = 10;
            let cols = 5;
            let data: Vec<f64> = vec![0.0; rows * cols];
            Ok((data, rows, cols))
        })?;

        // Convert to numpy array
        let (flat_data, rows, cols) = data;
        let array = PyArray2::from_vec2_bound(
            py,
            &flat_data
                .chunks(cols)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyException::new_err(format!("Failed to create numpy array: {}", e)))?;

        Ok(array)
    }

    /// Get storage configuration information
    fn get_config(&self) -> PyResult<String> {
        Ok(format!(
            "LanceStorage(output_dir='{}', max_rows_per_file={}, max_rows_per_group={}, compression='{}')",
            self.output_dir.display(),
            self.max_rows_per_file,
            self.max_rows_per_group,
            self.compression
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "LanceStorage(output_dir='{}')",
            self.output_dir.display()
        )
    }
}

/// Python module definition
#[pymodule]
#[pyo3(name = "genegraph")]
fn genegraph_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StorageBuilder>()?;
    m.add_class::<LanceStorage>()?;

    // Convenience function to create a storage builder
    #[pyfn(m)]
    #[pyo3(name = "create_storage")]
    fn create_storage(output_dir: String) -> PyResult<StorageBuilder> {
        Ok(StorageBuilder::new(output_dir, None, None, None))
    }

    Ok(())
}
