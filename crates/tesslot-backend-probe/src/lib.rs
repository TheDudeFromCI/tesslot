//! Opt-in numerical capability probes. Never a dependency of the engine or API.
//!
//! Run `cargo test -p tesslot-backend-probe --features burn-cpu` to evaluate the
//! pinned Burn release. A CPU device copy is not evidence of multi-GPU autograd.
//! Optional real CUDA gates: `scripts/test-cuda.sh`; they are ignored by default.

#[cfg(all(test, feature = "burn-cpu"))]
mod tests {
    use burn_tensor::{Bool, DType, Device, Tensor, activation::softmax};

    #[test]
    fn matmul_copy_backward_matches_analytic_gradients() {
        let device = Device::flex().autodiff();
        let x = Tensor::<2>::from_data([[1.0, 2.0]], (&device, DType::F64)).require_grad();
        let w = Tensor::<2>::from_data([[3.0], [4.0]], (&device, DType::F64)).require_grad();
        let y = x.clone().to_device(&device).matmul(w.clone());
        let grads = y.clone().powf_scalar(2.0).sum().backward();
        let dx = x
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        let dw = w
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        assert_eq!(y.into_data().try_to_vec::<f64>().unwrap(), vec![11.0]);
        assert_eq!(dx, vec![66.0, 88.0]);
        assert_eq!(dw, vec![22.0, 44.0]);
    }

    #[test]
    fn finite_difference_all_parameters_across_attached_chunk_boundary() {
        let device = Device::flex().autodiff();
        let a = Tensor::<1>::from_data([0.3, -0.7], (&device, DType::F64)).require_grad();
        let b = Tensor::<1>::from_data([0.4, 0.8], (&device, DType::F64)).require_grad();
        // Chunk one produces attached memory. Chunk two consumes it directly.
        let prefix = a.clone().tanh();
        let loss = (prefix * b.clone()).powf_scalar(2.0).sum();
        let grads = loss.backward();
        let da = a
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        let db = b
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        let f = |a: [f64; 2], b: [f64; 2]| {
            a.iter()
                .zip(b)
                .map(|(a, b)| (a.tanh() * b).powi(2))
                .sum::<f64>()
        };
        for i in 0..2 {
            let mut plus = [0.3, -0.7];
            let mut minus = plus;
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            assert!((da[i] - (f(plus, [0.4, 0.8]) - f(minus, [0.4, 0.8])) / 2e-6).abs() < 1e-8);
            let mut plus = [0.4, 0.8];
            let mut minus = plus;
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            assert!((db[i] - (f([0.3, -0.7], plus) - f([0.3, -0.7], minus)) / 2e-6).abs() < 1e-8);
        }
    }

    #[test]
    fn checkpoint_strategy_preserves_outputs_and_gradients() {
        let run = |checkpoint: bool| {
            let device = Device::flex().autodiff();
            let device = if checkpoint {
                device.gradient_checkpointing()
            } else {
                device
            };
            let x = Tensor::<1>::from_data([0.3, -0.7], (&device, DType::F64)).require_grad();
            let y = x.clone().tanh().exp().powf_scalar(2.0).sum();
            let grads = y.backward();
            (
                y.into_data().try_to_vec::<f64>().unwrap(),
                x.grad(&grads)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap(),
            )
        };
        let a = run(false);
        let b = run(true);
        for (a, b) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn explicit_allowed_mask_excludes_value_and_gradient() {
        let device = Device::flex().autodiff();
        let scores = Tensor::<2>::from_data([[0.2, 99.0]], (&device, DType::F64)).require_grad();
        let values = Tensor::<2>::from_data([[3.0, 999.0]], (&device, DType::F64)).require_grad();
        let allowed = Tensor::<2, Bool>::from_data([[true, false]], &device);
        // Export true means allowed; Burn mask_fill true means replace.
        let probabilities = softmax(scores.mask_fill(allowed.bool_not(), f64::NEG_INFINITY), 1);
        let result = (probabilities * values.clone()).sum();
        let grads = result.backward();
        assert_eq!(result.into_data().try_to_vec::<f64>().unwrap(), vec![3.0]);
        assert_eq!(
            values
                .grad(&grads)
                .unwrap()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![1.0, 0.0]
        );
    }
}
