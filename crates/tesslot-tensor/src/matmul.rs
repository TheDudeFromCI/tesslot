//! Strict FP32 reference GEMM composed of differentiable multiply/reduce operations.
//!
//! CubeCL 0.11.3 may use TF32 for an ordinary FP32 `Tensor::matmul`. This bounded-tile
//! implementation avoids GEMM dispatch in both forward and backward. It is intended
//! for numerical qualification, not production throughput or memory efficiency.
use burn_tensor::{DType, Tensor};

/// Compute `[m,k] × [k,n]` using FP32 arithmetic without TF32 operand rounding.
///
/// Each product tile contains at most 16³ values. Autograd still retains the
/// required tiles over the whole operation, so this is not an activation-memory
/// optimization. Inputs must be nonempty FP32 tensors on the same device.
pub fn reference_f32(lhs: Tensor<2>, rhs: Tensor<2>) -> Result<Tensor<2>, String> {
    let [m, k] = lhs.dims();
    let [rhs_k, n] = rhs.dims();
    if m == 0 || k == 0 || n == 0 || k != rhs_k {
        return Err("reference FP32 matmul requires nonempty compatible matrix dimensions".into());
    }
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        return Err("reference FP32 matmul requires FP32 operands".into());
    }
    if lhs.device() != rhs.device() {
        return Err("reference FP32 matmul requires operands on the same device".into());
    }
    const TILE: usize = 16;
    let mut rows = Vec::new();
    for i in (0..m).step_by(TILE) {
        let end_i = (i + TILE).min(m);
        let mut columns = Vec::new();
        for j in (0..n).step_by(TILE) {
            let end_j = (j + TILE).min(n);
            let mut sum: Option<Tensor<2>> = None;
            for start_k in (0..k).step_by(TILE) {
                let end_k = (start_k + TILE).min(k);
                let a = lhs
                    .clone()
                    .slice([i..end_i, start_k..end_k])
                    .unsqueeze_dim::<3>(2);
                let b = rhs
                    .clone()
                    .slice([start_k..end_k, j..end_j])
                    .unsqueeze_dim::<3>(0);
                let product = (a * b).sum_dim(1).reshape([end_i - i, end_j - j]);
                sum = Some(match sum {
                    Some(total) => total + product,
                    None => product,
                });
            }
            columns.push(sum.ok_or("empty reference FP32 reduction")?);
        }
        rows.push(Tensor::cat(columns, 1));
    }
    Ok(Tensor::cat(rows, 0))
}
