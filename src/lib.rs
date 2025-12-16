#![allow(unused_variables)]
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyValueError};
use std::path::PathBuf;
use genegraph_storage::lance_storage_graph::LanceStorageGraph;
use smartcore::linalg::basic::matrix::DenseMatrix;
use genegraph_storage::traits::backend::StorageBackend;
use genegraph_storage::traits::metadata::Metadata;


#[cfg(test)]
mod tests;


use std::sync::Once;
static INIT: Once = Once::new();

/// Initialize logging for tests
pub fn init() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
            .ok();
    });
}

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
    ///     2D numpy array to store (dense matrix). Must contain only finite values.
    /// name : str
    ///     Name of the dataset/file
    /// 
    /// Returns
    /// -------
    /// str
    ///     Path to the stored Lance dataset
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the array contains non-finite values (NaN, Inf, -Inf)
    #[pyo3(signature = (array, name))]
    fn store<'py>(
        &self,
        py: Python<'py>,
        array: PyReadonlyArray2<'py, f64>,
        name: String,
    ) -> PyResult<String> {
        // Extract data we need before releasing GIL
        let array_view = array.as_array();
        let shape = array_view.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        // Validate dimensions
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err(
                "Cannot store empty array. Array must have non-zero dimensions."
            ));
        }
        
        // Step 1: Efficiently check for non-finite values
        // Use iterator to enable early termination on first non-finite value
        let has_non_finite = array_view.iter().any(|&v| !v.is_finite());
        
        if has_non_finite {
            // Count total non-finite values for detailed error message
            let non_finite_count = array_view.iter().filter(|&&v| !v.is_finite()).count();
            let nan_count = array_view.iter().filter(|&&v| v.is_nan()).count();
            let inf_count = array_view.iter().filter(|&&v| v.is_infinite()).count();
            
            return Err(PyValueError::new_err(format!(
                "Array contains {} non-finite values ({} NaN, {} Inf/-Inf). \
                 Only arrays with finite values can be stored as embeddings. \
                 Please clean your data before storing.",
                non_finite_count, nan_count, inf_count
            )));
        }
        
        // Step 2: Convert to owned data (Vec) so we can safely pass to another thread
        let data: Vec<f64> = array_view.iter().copied().collect();
        
        // Step 3: Now release GIL and do async computation
        let output_dir = self.output_dir.clone();
        let compression = self.compression.clone();
        
        let output_path = py.detach(move || {
            // Use tokio runtime for async operations (required by genegraph-storage)
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| PyException::new_err(format!("Failed to create runtime: {}", e)))?;
            
            runtime.block_on(async move {
                // Construct output path
                let path = output_dir.join(format!("{}.lance", name));
                
                // Create LanceStorageGraph instance
                let storage = LanceStorageGraph::new(
                    output_dir.to_string_lossy().to_string(),
                    name.clone(),
                );
                
                // Convert Vec<f64> to DenseMatrix (column-major format for smartcore)
                // genegraph-storage expects column-major DenseMatrix
                let matrix = DenseMatrix::new(rows, cols, data, true)
                    .map_err(|e| PyException::new_err(format!("Failed to create DenseMatrix: {}", e)))?;
                
                // Create a minimal metadata file (required by genegraph-storage)
                let metadata_path = storage.metadata_path();
                if !metadata_path.exists() {
                    use genegraph_storage::metadata::GeneMetadata;
                    
                    let metadata = GeneMetadata::new(
                        &name
                    );
                    
                    storage.save_metadata(&metadata).await
                        .map_err(|e| PyException::new_err(format!("Failed to save metadata: {}", e)))?;
                }
                
                // Use genegraph-storage to write to Lance format
                storage.save_dense(&name, &matrix, &metadata_path).await
                    .map_err(|e| PyException::new_err(format!("Failed to save dense matrix: {}", e)))?;
                
                Ok::<_, PyErr>(path)
            })
        })?;

        Ok(output_path.to_string_lossy().to_string())
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
        let lance_path = PathBuf::from(&path);
        
        let (flat_data, rows, cols) = py.detach(move || {
            // TODO: Use genegraph-storage crate to read from Lance format
            // This is a placeholder for the actual implementation
            // 
            // use genegraph_storage::lance::LanceReader;
            // let reader = LanceReader::open(&lance_path)?;
            // let (data, rows, cols) = reader.read_dense_matrix()?;
            // Ok::<_, PyErr>((data, rows, cols))

            // Placeholder: return dummy data
            // Replace with actual genegraph-storage implementation
            let rows = 10;
            let cols = 5;
            let data: Vec<f64> = vec![0.0; rows * cols];
            Ok::<_, PyErr>((data, rows, cols))
        })?;

        // Convert flat data to 2D vec
        let vec2d: Vec<Vec<f64>> = flat_data
            .chunks(cols)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Create numpy array using the bounded interface
        let array = PyArray2::from_vec2(py, &vec2d)
            .map_err(|e| PyException::new_err(format!("Failed to create numpy array: {}", e)))?;

        Ok(array)
    }

    /// Store a batch of numpy arrays to Lance format
    /// 
    /// Parameters
    /// ----------
    /// arrays : list[numpy.ndarray]
    ///     List of 2D numpy arrays to store
    /// names : list[str]
    ///     Names for each dataset
    /// 
    /// Returns
    /// -------
    /// list[str]
    ///     Paths to the stored Lance datasets
    #[pyo3(signature = (arrays, names))]
    fn store_batch<'py>(
        &self,
        py: Python<'py>,
        arrays: Vec<PyReadonlyArray2<'py, f64>>,
        names: Vec<String>,
    ) -> PyResult<Vec<String>> {
        if arrays.len() != names.len() {
            return Err(PyException::new_err(
                "Number of arrays must match number of names"
            ));
        }

        let mut results = Vec::new();
        for (array, name) in arrays.iter().zip(names.iter()) {
            let path = self.store(py, array.clone(), name.clone())?;
            results.push(path);
        }

        Ok(results)
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

    /// Get the output directory path
    fn get_output_dir(&self) -> String {
        self.output_dir.to_string_lossy().to_string()
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
fn genestore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StorageBuilder>()?;
    m.add_class::<LanceStorage>()?;

    // Module version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Convenience function to create a storage builder
    #[pyfn(m)]
    #[pyo3(name = "create_storage")]
    fn create_storage(output_dir: String) -> PyResult<StorageBuilder> {
        Ok(StorageBuilder::new(output_dir, None, None, None))
    }

    Ok(())
}
