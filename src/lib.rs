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
use short_uuid::ShortUuid;
use log::{info, debug};

use std::sync::Once;
static INIT: Once = Once::new();

/// Initialize logging for tests
pub fn init() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
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
            let mut new_storage = true;

            let (check_exist, _) = LanceStorageGraph::exists(&output_dir.to_string_lossy().to_string());
            let (storage, mut metadata)  = if !check_exist { 
                info!("creating new storage at {:?}", output_dir);
                // new storage
                let storage = LanceStorageGraph::new(
                    output_dir.to_string_lossy().to_string(),
                    name.to_string(),
                );
                let md = GeneMetadata::seed_metadata(&name, rows, cols, &storage)
                    .await
                    .unwrap();

                (storage, md)
            } else {
                // spawn storage
                info!("spawning storage at {:?}", output_dir);
                new_storage = false;
                LanceStorageGraph::spawn(output_dir.to_string_lossy().to_string())
                    .await
                    .map_err(|e| PyException::new_err(format!("Failed to spawn storage (metadata missing?): {}", e)))?
            };

            // Data is now in column-major format
            let matrix = DenseMatrix::new(rows, cols, data_col_major, true)
                .map_err(|e| PyException::new_err(format!("Failed to create DenseMatrix: {}", e)))?;

            let metadata_path = storage.metadata_path();
            
            let metadata = if new_storage {               
                // Save the dense matrix
                storage.save_dense("rawinput", &matrix, &metadata_path).await
                    .map_err(|e| PyException::new_err(format!("Failed to save dense matrix: {}", e)))?;
            
                metadata
            } else {
                // Add file info to metadata
                metadata.files.insert(
                    name.to_string(),
                    metadata.new_fileinfo(
                        &name,
                        "dense",  // filetype
                        matrix.shape(),
                        None,
                        None,
                    ),
                );
                
                {
                    // Save updated metadata
                    storage.save_metadata(&metadata).await
                        .map_err(|e| PyException::new_err(format!("Failed to save metadata: {}", e)))?;

                    // Save the dense matrix
                    storage.save_dense(&name, &matrix, &metadata_path).await
                        .map_err(|e| PyException::new_err(format!("Failed to save dense matrix: {}", e)))?;
                }

                storage.load_metadata().await
                    .map_err(|e| PyException::new_err(format!("Failed to load existing metadata: {}", e)))?
            };


            Ok(metadata_path.to_string_lossy().to_string())
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
            let (storage, metadata) = LanceStorageGraph::spawn(output_dir.to_string_lossy().to_string())
                .await
                .map_err(|e| PyException::new_err(format!("Failed to spawn storage (metadata missing?): {}", e)))?;
            
            // Load the dense matrix using the dataset name or load the root rawinput
            // TODO: use map_error and return an error
            let (fileinfo, key) = if metadata.files.get(&name).is_some() {
                (metadata.files.get(&name).unwrap(), name.clone())
            } else {
                   (metadata.files.get("rawinput").unwrap(), "rawinput".to_string()) 
            };
            let matrix = storage.load_dense(&key).await
                .map_err(|e| PyException::new_err(format!("Failed to load dense matrix '{}': {}", &key, e)))?;
            
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

    /// Each directory is a separate storage.
    /// If the same directory is passed, arrays are stored in the same storage
    #[pyfn(m)]
    #[pyo3(name = "store_array")]
    fn store_array(output_dir: String) -> PyResult<StorageBuilder> {
        Ok(StorageBuilder::new(output_dir, None, None, None))
    }

    Ok(())
}
