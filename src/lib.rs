#![allow(unused_variables)]

use genegraph_storage::lance_storage_graph::LanceStorageGraph;
use genegraph_storage::metadata::GeneMetadata;
use genegraph_storage::traits::backend::StorageBackend;
use genegraph_storage::traits::metadata::Metadata;

use log::{debug, info};
use numpy::{PyArray2, PyArrayLike2, PyReadonlyArray2};
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::{future_into_py, get_runtime};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use std::path::PathBuf;
use std::sync::{Arc, Once};

static INIT: Once = Once::new();

/// Initialize logging for tests
pub fn init() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .try_init()
            .ok();
    });
}

/// Shared storage config/state so sync + async wrappers can clone it safely.
#[derive(Clone)]
struct LanceStorageInner {
    output_dir: PathBuf,
    max_rows_per_file: usize,
    max_rows_per_group: usize,
    compression: String,
}

/// Helper: validate array and convert numpy row-major -> col-major Vec<f64>.
fn numpy_to_col_major_from_view(
    array_view: &ndarray::ArrayView2<'_, f64>,
) -> PyResult<(usize, usize, Vec<f64>)> {
    let shape = array_view.shape();
    let rows = shape[0];
    let cols = shape[1];

    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err(
            "Cannot store empty array. Array must have non-zero dimensions.",
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

    // Convert row-major to column-major
    let mut data_col_major: Vec<f64> = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data_col_major[c * rows + r] = array_view[[r, c]];
        }
    }

    Ok((rows, cols, data_col_major))
}

impl LanceStorageInner {
    async fn store_core(
        &self,
        name: String,
        rows: usize,
        cols: usize,
        data_col_major: Vec<f64>,
    ) -> PyResult<String> {
        let output_dir = self.output_dir.clone();

        let (check_exist, _) = LanceStorageGraph::exists(&output_dir.to_string_lossy().to_string());

        let (storage, metadata) = if !check_exist {
            info!("creating new storage at {:?}", output_dir);
            let storage =
                LanceStorageGraph::new(output_dir.to_string_lossy().to_string(), name.to_string());

            let md = GeneMetadata::seed_metadata(&name, rows, cols, &storage)
                .await
                .map_err(|e| PyException::new_err(format!("Failed to seed metadata: {}", e)))?;

            (storage, md)
        } else {
            info!("spawning storage at {:?}", output_dir);
             LanceStorageGraph::spawn(output_dir.to_string_lossy().to_string())
                .await
                .map_err(|e| {
                    PyException::new_err(format!(
                        "Failed to spawn storage (metadata missing?): {}",
                        e
                    ))
                })?
        };

        let matrix = DenseMatrix::new(rows, cols, data_col_major, true)
            .map_err(|e| PyException::new_err(format!("Failed to create DenseMatrix: {}", e)))?;

        let metadata_path = storage.metadata_path();

        storage
            .save_dense(&name, &matrix, &metadata_path)
            .await
            .map_err(|e| PyException::new_err(format!("Failed to save dense matrix: {}", e)))?;

        Ok(metadata_path.to_string_lossy().to_string())
    }

    async fn load_core(&self, name: String) -> PyResult<Vec<Vec<f64>>> {
        let output_dir = self.output_dir.clone();

        let (storage, metadata) =
            LanceStorageGraph::spawn(output_dir.to_string_lossy().to_string())
                .await
                .map_err(|e| {
                    PyException::new_err(format!(
                        "Failed to spawn storage (metadata missing?): {}",
                        e
                    ))
                })?;

        let key = if metadata.files.get(&name).is_some() {
            name.clone()
        } else {
            "rawinput".to_string()
        };

        let matrix = storage.load_dense(&key).await.map_err(|e| {
            PyException::new_err(format!("Failed to load dense matrix '{}': {}", &key, e))
        })?;

        let (rows, cols) = matrix.shape();

        // Build row-major Vec<Vec<f64>> for numpy conversion under GIL
        let mut vec2d: Vec<Vec<f64>> = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(*matrix.get((r, c)));
            }
            vec2d.push(row);
        }

        Ok(vec2d)
    }
}

/// Storage builder for configuring Lance storage parameters
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

    fn with_output_dir(&mut self, output_dir: String) -> PyResult<()> {
        self.output_dir = PathBuf::from(output_dir);
        Ok(())
    }

    fn with_max_rows_per_file(&mut self, max_rows: usize) -> PyResult<()> {
        self.max_rows_per_file = Some(max_rows);
        Ok(())
    }

    fn with_max_rows_per_group(&mut self, max_rows: usize) -> PyResult<()> {
        self.max_rows_per_group = Some(max_rows);
        Ok(())
    }

    fn with_compression(&mut self, compression: String) -> PyResult<()> {
        self.compression = Some(compression);
        Ok(())
    }

    fn build(&self) -> PyResult<LanceStorage> {
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| PyException::new_err(format!("Failed to create directory: {}", e)))?;

        Ok(LanceStorage {
            inner: Arc::new(LanceStorageInner {
                output_dir: self.output_dir.clone(),
                max_rows_per_file: self.max_rows_per_file.unwrap_or(1_000_000),
                max_rows_per_group: self.max_rows_per_group.unwrap_or(10_000),
                compression: self
                    .compression
                    .clone()
                    .unwrap_or_else(|| "zstd".to_string()),
            }),
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

/// Async facade exposed at `storage.aio`
#[pyclass]
pub struct LanceStorageAsync {
    inner: Arc<LanceStorageInner>,
}

#[pymethods]
impl LanceStorageAsync {
    #[pyo3(signature = (array, name))]
    fn store<'py>(
        &self,
        py: Python<'py>,
        array: PyArrayLike2<'_, f64>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let readonly_array = array.as_array();
        let (rows, cols, data_col_major) = numpy_to_col_major_from_view(&readonly_array)?;

        future_into_py(py, async move {
            inner.store_core(name, rows, cols, data_col_major).await
        })
    }

    #[pyo3(signature = (name))]
    fn load<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();

        future_into_py(py, async move {
            let vec2d = inner.load_core(name).await?;
            Python::attach(|py| {
                PyArray2::from_vec2(py, &vec2d)
                    .map(|arr| arr.into_any().unbind())
                    .map_err(|e| {
                        PyException::new_err(format!("Failed to create numpy array: {}", e))
                    })
            })
        })
    }

    fn __repr__(&self) -> String {
        "LanceStorageAsync()".to_string()
    }
}

/// Main storage interface for Lance operations
#[pyclass]
pub struct LanceStorage {
    inner: Arc<LanceStorageInner>,
}

#[pymethods]
impl LanceStorage {
    // --- Async API (unchanged behavior for your tests) ---

    #[pyo3(signature = (array, name))]
    fn store_async<'py>(
        &self,
        py: Python<'py>,
        array: PyArrayLike2<'_, f64>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let readonly_array = array.as_array();
        let (rows, cols, data_col_major) = numpy_to_col_major_from_view(&readonly_array)?;

        future_into_py(py, async move {
            inner.store_core(name, rows, cols, data_col_major).await
        })
    }

    #[pyo3(signature = (name))]
    fn load_async<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();

        future_into_py(py, async move {
            let vec2d = inner.load_core(name).await?;
            Python::attach(|py| {
                PyArray2::from_vec2(py, &vec2d)
                    .map(|arr| arr.into_any().unbind())
                    .map_err(|e| {
                        PyException::new_err(format!("Failed to create numpy array: {}", e))
                    })
            })
        })
    }

    // --- Blocking API (new) ---

    #[pyo3(signature = (array, name))]
    fn store(
        &self,
        py: Python<'_>,
        array: PyArrayLike2<'_, f64>, // Changed from PyReadonlyArray2
        name: String,
    ) -> PyResult<String> {
        let inner = self.inner.clone();
        let readonly_array = array.as_array(); // Convert to readonly array
        let (rows, cols, data_col_major) = numpy_to_col_major_from_view(&readonly_array)?;

        py.detach(|| {
            get_runtime()
                .block_on(async move { inner.store_core(name, rows, cols, data_col_major).await })
        })
    }

    #[pyo3(signature = (name))]
    fn load(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();

        let vec2d =
            py.detach(|| get_runtime().block_on(async move { inner.load_core(name).await }))?;

        PyArray2::from_vec2(py, &vec2d)
            .map(|arr| arr.into_any().unbind())
            .map_err(|e| PyException::new_err(format!("Failed to create numpy array: {}", e)))
    }

    // --- Optional async namespace accessor ---

    #[getter]
    fn aio(&self) -> LanceStorageAsync {
        LanceStorageAsync {
            inner: self.inner.clone(),
        }
    }

    // --- Existing helpers used by tests ---

    fn get_config(&self) -> PyResult<String> {
        Ok(format!(
            "LanceStorage(output_dir='{}', max_rows_per_file={}, max_rows_per_group={}, compression='{}')",
            self.inner.output_dir.display(),
            self.inner.max_rows_per_file,
            self.inner.max_rows_per_group,
            self.inner.compression
        ))
    }

    fn get_output_dir(&self) -> String {
        self.inner.output_dir.to_string_lossy().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "LanceStorage(output_dir='{}')",
            self.inner.output_dir.display()
        )
    }
}

/// Python module definition
#[pymodule]
fn genestore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init();

    m.add_class::<StorageBuilder>()?;
    m.add_class::<LanceStorage>()?;
    m.add_class::<LanceStorageAsync>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    /// Each directory is a separate storage.
    /// If the same directory is passed, arrays are stored in the same storage.
    #[pyfn(m)]
    #[pyo3(name = "store_array")]
    fn store_array(output_dir: String) -> PyResult<StorageBuilder> {
        Ok(StorageBuilder::new(output_dir, None, None, None))
    }

    Ok(())
}
