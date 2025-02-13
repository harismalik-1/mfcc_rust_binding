#![allow(non_local_definitions)]

#[doc = include_str!("../README.md")]
mod freqs;
pub mod mfcc;
mod ringbuffer;

pub use crate::mfcc::*;

use pyo3::prelude::*;

#[pymodule]
fn mfcc_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTransform>()?;
    Ok(())
}

#[pyclass]
struct PyTransform {
    inner: crate::mfcc::Transform,
}

#[pymethods]
impl PyTransform {
    #[new]
    fn new(sample_rate: usize, buffer_size: usize) -> Self {
        PyTransform {
            inner: crate::mfcc::Transform::new(sample_rate, buffer_size)
        }
    }

    #[pyo3(name = "transform")]
    fn py_transform(&mut self, input: Vec<i16>) -> PyResult<Vec<f64>> {
        let mut output = vec![0.0; self.inner.output_length()];
        self.inner.transform(&input, &mut output);
        Ok(output)
    }

    fn nfilters(&mut self, maxfilter: usize, nfilters: usize) -> PyResult<()> {
        self.inner.set_nfilters(maxfilter, nfilters);
        Ok(())
    }

    fn normlength(&mut self, length: usize) -> PyResult<()> {
        self.inner.set_normlength(length);
        Ok(())
    }
}
