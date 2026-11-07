//! CPU-verifiable oracle for the strict FP32 CUDA reference path.
use burn_tensor::{DType, Device, Tensor, TensorData};
use tesslot_tensor::matmul::reference_f32;
fn flat<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec().unwrap()
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(a) - b).abs() < 5e-5 * (1.0 + b.abs()),
            "coordinate {i}: {a} vs {b}"
        );
    }
}
#[test]
fn rectangular_strided_tiles_and_both_gradients_match_scalar_oracle() {
    let (m, k, n) = (33, 19, 21);
    let device = Device::flex().autodiff();
    let a = (0..m * k)
        .map(|i| ((i * 19) % 31) as f32 / 17.0 - 0.73)
        .collect::<Vec<_>>();
    let b = (0..k * n)
        .map(|i| ((i * 7) % 29) as f32 / 23.0 - 0.31)
        .collect::<Vec<_>>();
    // A noncontiguous transposed leaf, dimensions all cross tile boundaries.
    let lhs = Tensor::<2>::from_data(TensorData::new(a.clone(), [k, m]), (&device, DType::F32))
        .transpose()
        .detach()
        .require_grad();
    let rhs = Tensor::<2>::from_data(TensorData::new(b.clone(), [k, n]), (&device, DType::F32))
        .require_grad();
    let result = reference_f32(lhs.clone(), rhs.clone()).unwrap();
    let mut oracle = vec![0.0; m * n];
    let mut da = vec![0.0; m * k];
    let mut db = vec![0.0; k * n];
    for i in 0..m {
        for j in 0..n {
            for t in 0..k {
                oracle[i * n + j] += f64::from(a[t * m + i]) * f64::from(b[t * n + j]);
            }
        }
    }
    for i in 0..m {
        for j in 0..n {
            for t in 0..k {
                da[i * k + t] += 2.0 * oracle[i * n + j] * f64::from(b[t * n + j]);
                db[t * n + j] += 2.0 * oracle[i * n + j] * f64::from(a[t * m + i]);
            }
        }
    }
    close(&flat(result.clone()), &oracle);
    let gradients = result.powf_scalar(2.0).sum().backward();
    close(&flat(lhs.grad(&gradients).unwrap()), &da);
    close(&flat(rhs.grad(&gradients).unwrap()), &db);
}
#[test]
fn workstation_failure_is_exactly_tf32_rounding_and_strict_mode_keeps_fp32() {
    // CubeCL's NVPTX backend uses cvt.rna.tf32.f32 (ties away from zero).
    let round = |x: f32| f32::from_bits((x.to_bits() + 0x1000) & !0x1fff);
    let tf32 = round(2.04) * round(1.016) + round(5.96) * round(-0.952);
    assert!((tf32 * tf32 - 12.994399).abs() < 1e-6);
    let device = Device::flex();
    let output = reference_f32(
        Tensor::from_data([[2.04, 5.96]], (&device, DType::F32)),
        Tensor::from_data([[1.016], [-0.952]], (&device, DType::F32)),
    )
    .unwrap();
    let strict = flat(output)[0];
    assert!((strict * strict - 12.969219).abs() < 2e-6);
    assert!(
        (strict * strict - tf32 * tf32).abs() > 0.02,
        "strict FP32 must not accept the TF32-rounded result"
    );
}
#[test]
fn reference_matmul_rejects_invalid_shapes_and_dtypes() {
    let device = Device::flex();
    let a = Tensor::<2>::from_data([[1.0, 2.0]], (&device, DType::F32));
    assert!(
        reference_f32(a.clone(), a.clone())
            .unwrap_err()
            .contains("dimensions")
    );
    let b = Tensor::<2>::from_data([[1.0], [2.0]], (&device, DType::F64));
    assert!(reference_f32(a, b).unwrap_err().contains("FP32 operands"));
}
