//! Opt-in hardware gates. Ignored by default, including when CUDA is compiled.
#![cfg(feature = "cuda")]
use burn_tensor::{DType, Device, Tensor};
use std::process::Command;
fn devices() -> usize {
    let Ok(output) = Command::new("nvidia-smi")
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
fn run(first: Device, second: Device) -> Vec<f32> {
    let x = Tensor::<2>::from_data([[1.0, 2.0]], (&first, DType::F32));
    let a = Tensor::<2>::from_data([[2.0, 0.0], [0.0, 3.0]], (&first, DType::F32)).require_grad();
    let b = Tensor::<2>::from_data([[1.0], [-1.0]], (&second, DType::F32)).require_grad();
    let y = x
        .clone()
        .matmul(a.clone())
        .to_device(&second)
        .matmul(b.clone());
    let loss = y.powf_scalar(2.0).sum();
    let grads = loss.backward();
    let da = a
        .grad(&grads)
        .expect("gradient must propagate to first device");
    let db = b
        .grad(&grads)
        .expect("gradient must reach second partition");
    let mut result = loss.into_data().try_to_vec::<f32>().unwrap();
    result.extend(da.clone().into_data().try_to_vec::<f32>().unwrap());
    result.extend(db.clone().into_data().try_to_vec::<f32>().unwrap());
    // Update both partitions and compare the next forward with CPU.
    let a = a.detach() - da * 0.001;
    let b = b.detach() - db * 0.001;
    let next = x
        .matmul(a)
        .to_device(&second)
        .matmul(b)
        .powf_scalar(2.0)
        .sum();
    result.extend(next.into_data().try_to_vec::<f32>().unwrap());
    result
}
fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - e).abs() <= 1e-4 * (1.0 + e.abs()),
            "CUDA {a} differs from CPU {e}"
        );
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
    compare(&run(cuda.clone(), cuda), &expected);
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
    compare(
        &run(Device::cuda(0).autodiff(), Device::cuda(1).autodiff()),
        &expected,
    );
}
