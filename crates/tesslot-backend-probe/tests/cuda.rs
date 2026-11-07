//! Stage-labelled FP32 transfer/gradient gates. Hardware cases are opt-in.
#![cfg(feature = "burn-cpu")]
use burn_tensor::{DType, Device, Tensor};
use tesslot_tensor::matmul::reference_f32;

#[derive(Debug)]
struct Snapshot {
    loss: Vec<f32>,
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
    updated_a: Vec<f32>,
    updated_b: Vec<f32>,
    updated_hidden: Vec<f32>,
    updated_output: Vec<f32>,
    updated_loss: Vec<f32>,
}
impl Snapshot {
    fn stages(&self) -> [(&str, &[f32]); 8] {
        [
            ("loss", &self.loss),
            ("gradient_a", &self.gradient_a),
            ("gradient_b", &self.gradient_b),
            ("updated_a", &self.updated_a),
            ("updated_b", &self.updated_b),
            ("updated_hidden", &self.updated_hidden),
            ("updated_output", &self.updated_output),
            ("updated_loss", &self.updated_loss),
        ]
    }
}
fn flat<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}
fn run(first: Device, second: Device) -> Snapshot {
    let x = Tensor::<2>::from_data([[1.0, 2.0]], (&first, DType::F32));
    let a = Tensor::<2>::from_data([[2.0, 0.0], [0.0, 3.0]], (&first, DType::F32)).require_grad();
    let b = Tensor::<2>::from_data([[1.0], [-1.0]], (&second, DType::F32)).require_grad();
    let hidden = reference_f32(x.clone(), a.clone())
        .unwrap()
        .to_device(&second);
    let y = reference_f32(hidden, b.clone()).unwrap();
    let loss = y.powf_scalar(2.0).sum();
    let grads = loss.backward();
    let da = a
        .grad(&grads)
        .expect("gradient must propagate to first device");
    let db = b
        .grad(&grads)
        .expect("gradient must reach second partition");
    let next_a = a.detach() - da.clone() * 0.001;
    let next_b = b.detach() - db.clone() * 0.001;
    let hidden = reference_f32(x, next_a.clone()).unwrap().to_device(&second);
    let output = reference_f32(hidden.clone(), next_b.clone()).unwrap();
    let next_loss = output.clone().powf_scalar(2.0).sum();
    Snapshot {
        loss: flat(loss),
        gradient_a: flat(da),
        gradient_b: flat(db),
        updated_a: flat(next_a),
        updated_b: flat(next_b),
        updated_hidden: flat(hidden),
        updated_output: flat(output),
        updated_loss: flat(next_loss),
    }
}
fn compare(actual: &Snapshot, expected: &Snapshot) {
    for ((name, actual), (_, expected)) in actual.stages().into_iter().zip(expected.stages()) {
        assert_eq!(actual.len(), expected.len(), "{name} shape");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (a - e).abs() <= 1e-4 * (1.0 + e.abs()),
                "strict FP32 {name}[{index}]: actual={a}, expected={e}; full actual={actual:?}"
            );
        }
    }
}
#[test]
fn cpu_probe_matches_independent_analytic_update() {
    let cpu = Device::flex().autodiff();
    compare(
        &run(cpu.clone(), cpu),
        &Snapshot {
            loss: vec![16.0],
            gradient_a: vec![-8.0, 8.0, -16.0, 16.0],
            gradient_b: vec![-16.0, -48.0],
            updated_a: vec![2.008, -0.008, 0.016, 2.984],
            updated_b: vec![1.016, -0.952],
            updated_hidden: vec![2.04, 5.96],
            updated_output: vec![-3.60128],
            updated_loss: vec![12.969219],
        },
    );
}

#[cfg(feature = "cuda")]
mod hardware {
    use super::*;
    fn devices() -> usize {
        let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index", "--format=csv,noheader"])
            .output()
        else {
            return 0;
        };
        if !output.status.success() {
            return 0;
        }
        let physical = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|s| !s.trim().is_empty())
            .count();
        match std::env::var("CUDA_VISIBLE_DEVICES") {
            Ok(v) if v.trim().is_empty() || v.trim() == "-1" => 0,
            Ok(v) => physical.min(v.split(',').filter(|s| !s.trim().is_empty()).count()),
            Err(_) => physical,
        }
    }
    #[test]
    #[ignore = "requires NVIDIA CUDA driver/compiler; run scripts/test-cuda.sh"]
    fn single_cuda_forward_backward_update_matches_cpu() {
        if devices() < 1 {
            eprintln!("SKIP single CUDA gate: no visible NVIDIA GPU");
            return;
        }
        let cpu = Device::flex().autodiff();
        let expected = run(cpu.clone(), cpu);
        let cuda = Device::cuda(0).autodiff();
        let actual = run(cuda.clone(), cuda);
        eprintln!("single CUDA strict FP32 stages: {actual:#?}");
        compare(&actual, &expected);
    }
    #[test]
    #[ignore = "requires two visible NVIDIA CUDA GPUs; run scripts/test-cuda.sh"]
    fn dual_cuda_transfer_backward_update_matches_cpu() {
        if devices() < 2 {
            eprintln!("SKIP dual CUDA gate: fewer than two visible NVIDIA GPUs");
            return;
        }
        let cpu = Device::flex().autodiff();
        let expected = run(cpu.clone(), cpu);
        let actual = run(Device::cuda(0).autodiff(), Device::cuda(1).autodiff());
        eprintln!("dual CUDA strict FP32 stages: {actual:#?}");
        compare(&actual, &expected);
    }
    #[test]
    #[ignore = "diagnose backend-native FP32 GEMM, which can round operands to TF32"]
    fn native_fp32_matmul_reports_tf32_or_simt_rounding() {
        if devices() < 1 {
            eprintln!("SKIP native TF32 diagnostic: no visible NVIDIA GPU");
            return;
        }
        let device = Device::cuda(0);
        let a = Tensor::<2>::from_data([[2.04, 5.96]], (&device, DType::F32));
        let b = Tensor::<2>::from_data([[1.016], [-0.952]], (&device, DType::F32));
        let actual = flat(a.matmul(b))[0];
        // Both are legitimate backend-native outcomes, depending on kernel/hardware selection.
        // The strict FP32 gates above accept only the independent full-precision oracle.
        let fp32 = -3.60128f32;
        let tf32 = -3.6047745f32;
        assert!(
            actual.is_finite() && ((actual - fp32).abs() < 1e-5 || (actual - tf32).abs() < 1e-5),
            "native GEMM {actual}: neither FP32 {fp32} nor TF32 {tf32}"
        );
        eprintln!("native FP32-storage GEMM: {actual}; strict FP32={fp32}; TF32-rounded={tf32}");
    }
}
