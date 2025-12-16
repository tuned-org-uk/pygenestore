#![allow(unused_variables)]
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3_async_runtimes::tokio::future_into_py;
use std::path::PathBuf;
use genegraph_storage::lance_storage_graph::LanceStorageGraph;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use genegraph_storage::traits::backend::StorageBackend;
use genegraph_storage::traits::metadata::Metadata;
use genegraph_storage::metadata::GeneMetadata;



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
   /// Store a numpy array to Lance format (async)
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
    ///     Path to the stored Lance dataset (awaitable)
    #[pyo3(signature = (array, name))]
    fn store<'py>(
        &self,
        py: Python<'py>,
        array: PyReadonlyArray2<'py, f64>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Validate and extract data BEFORE entering the async block
        let array_view = array.as_array();
        let shape = array_view.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err(
                "Cannot store empty array. Array must have non-zero dimensions."
            ));
        }
        
        let has_non_finite = array_view.iter().any(|&v| !v.is_finite());
        
        if has_non_finite {
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
        
        // Convert numpy row-major to column-major (smartcore/genegraph-storage convention)
        let mut data_col_major: Vec<f64> = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let col_major_idx = c * rows + r;
                data_col_major[col_major_idx] = array_view[[r, c]];
            }
        }
        
        let output_dir = self.output_dir.clone();
        
        // Return a Python awaitable
        future_into_py(py, async move {
            let storage = LanceStorageGraph::new(
                output_dir.to_string_lossy().to_string(),
                name.clone(),
            );
            
            // Data is now in column-major format
            let matrix = DenseMatrix::new(rows, cols, data_col_major, true)
                .map_err(|e| PyException::new_err(format!("Failed to create DenseMatrix: {}", e)))?;
            
            let metadata_path = storage.metadata_path();
            let md = GeneMetadata::seed_metadata(&name, rows, cols, &storage.clone())
                .await
                .unwrap();
            
            storage.save_dense("rawinput", &matrix, &metadata_path).await
                .map_err(|e| PyException::new_err(format!("Failed to save dense matrix: {}", e)))?;
            
            let path = output_dir.join(format!("{}_data.lance", name));
            Ok(path.to_string_lossy().to_string())
        })
    }

    /// Load a Lance dataset into a numpy array (async)
    /// 
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the dataset (same as used in store())
    /// 
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     2D numpy array containing the loaded data (awaitable)
    #[pyo3(signature = (name))]
    fn load<'py>(
        &self,
        py: Python<'py>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let output_dir = self.output_dir.clone();
        
        future_into_py(py, async move {
            // Use spawn to read metadata and initialize storage from existing directory
            let (storage, _metadata) = LanceStorageGraph::spawn(output_dir.to_string_lossy().to_string())
                .await
                .map_err(|e| PyException::new_err(format!("Failed to spawn storage (metadata missing?): {}", e)))?;
            
            // Load the dense matrix using the dataset name
            let matrix = storage.load_dense("rawinput").await
                .map_err(|e| PyException::new_err(format!("Failed to load dense matrix '{}': {}", name, e)))?;
            
            let (rows, cols) = matrix.shape();
            
            // Extract in row-major order for numpy
            let mut flat_row_major = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    flat_row_major.push(*matrix.get((r, c)));
                }
            }
            
            // Convert to 2D vec
            let vec2d: Vec<Vec<f64>> = flat_row_major
                .chunks(cols)
                .map(|chunk| chunk.to_vec())
                .collect();
            
            // Create numpy array (must be done inside GIL)
            Python::attach(|py| {
                PyArray2::from_vec2(py, &vec2d)
                    .map(|arr| arr.into_any().unbind())
                    .map_err(|e| PyException::new_err(format!("Failed to create numpy array: {}", e)))
            })
        })
    }

    /// Store a batch of numpy arrays to Lance format (async)
    #[pyo3(signature = (arrays, names))]
    fn store_batch<'py>(
        &self,
        py: Python<'py>,
        arrays: Vec<PyReadonlyArray2<'py, f64>>,
        names: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if arrays.len() != names.len() {
            return Err(PyValueError::new_err(
                "Number of arrays must match number of names",
            ));
        }

        // Clone needed fields from self to avoid lifetime issues
        let output_dir = self.output_dir.clone();
        
        // Validate and extract all arrays upfront
        let mut batch_data = Vec::with_capacity(arrays.len());
        for (arr, name) in arrays.into_iter().zip(names.iter()) {
            let view = arr.as_array();
            let shape = view.shape();
            let rows = shape[0];
            let cols = shape[1];

            if rows == 0 || cols == 0 {
                return Err(PyValueError::new_err(
                    "Cannot store empty array in batch.",
                ));
            }
            
            if view.iter().any(|&v| !v.is_finite()) {
                return Err(PyValueError::new_err(
                    "Batch contains non-finite values.",
                ));
            }

            // Convert to column-major
            let mut data_col_major = vec![0.0; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    data_col_major[c * rows + r] = view[[r, c]];
                }
            }
            
            batch_data.push((name.clone(), rows, cols, data_col_major));
        }

        future_into_py(py, async move {
            let mut paths = Vec::new();
            
            for (name, rows, cols, data) in batch_data {
                let path = output_dir.join(format!("{}.lance", name));
                
                let storage = LanceStorageGraph::new(
                    output_dir.to_string_lossy().to_string(),
                    name.clone(),
                );

                let matrix = DenseMatrix::new(rows, cols, data, true)
                    .map_err(|e| PyException::new_err(format!("Failed to create DenseMatrix: {}", e)))?;

                let md_path = storage.metadata_path();
                if !md_path.exists() {
                    use genegraph_storage::metadata::GeneMetadata;
                    let md = GeneMetadata::new(&name);
                    storage.save_metadata(&md).await
                        .map_err(|e| PyException::new_err(format!("Failed to save metadata: {}", e)))?;
                }

                storage.save_dense(&name, &matrix, &md_path).await
                    .map_err(|e| PyException::new_err(format!("Failed to save dense: {}", e)))?;

                paths.push(path.to_string_lossy().to_string());
            }
            
            Ok(paths)
        })
    }

    // Keep the synchronous helper methods
    fn get_config(&self) -> PyResult<String> {
        Ok(format!(
            "LanceStorage(output_dir='{}', max_rows_per_file={}, max_rows_per_group={}, compression='{}')",
            self.output_dir.display(),
            self.max_rows_per_file,
            self.max_rows_per_group,
            self.compression
        ))
    }

    fn get_output_dir(&self) -> String {
        self.output_dir.to_string_lossy().to_string()
    }

    fn __repr__(&self) -> String {
        format!("LanceStorage(output_dir='{}')", self.output_dir.display())
    }
}

/// Python module definition
#[pymodule]
fn genestore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init();
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
