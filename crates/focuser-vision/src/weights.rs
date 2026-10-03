//! Reading model weights out of a safetensors file into Burn tensors.

use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor, TensorData};
use safetensors::{Dtype, SafeTensors};

use crate::{Result, VisionError};

/// A safetensors file held in memory. Float16 and float32 are both accepted;
/// the GPU computes in float32 either way.
pub struct Weights {
    bytes: Vec<u8>,
}

impl Weights {
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        // Parse once up front so a corrupt file fails here, not mid-load.
        SafeTensors::deserialize(&bytes).map_err(|e| VisionError::Model(e.to_string()))?;
        Ok(Self { bytes })
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        Self::new(std::fs::read(path)?)
    }

    pub(crate) fn floats(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let st =
            SafeTensors::deserialize(&self.bytes).map_err(|e| VisionError::Model(e.to_string()))?;
        let view = st
            .tensor(name)
            .map_err(|_| VisionError::Model(format!("weights are missing {name}")))?;
        let data = view.data();
        let floats = match view.dtype() {
            Dtype::F32 => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::F16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            other => {
                return Err(VisionError::Model(format!(
                    "{name}: unsupported dtype {other:?}"
                )));
            }
        };
        Ok((floats, view.shape().to_vec()))
    }

    pub fn tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Result<Tensor<B, D>> {
        let (floats, shape) = self.floats(name)?;
        let shape: [usize; D] = shape.try_into().map_err(|s: Vec<usize>| {
            VisionError::Model(format!("{name}: expected {D} dims, got {s:?}"))
        })?;
        Ok(Tensor::from_data(TensorData::new(floats, shape), device))
    }
}
