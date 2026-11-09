//! Activation stability at magnitudes reachable with pretrained weights.
use super::*;

fn verify(device: Device) {
    let values = vec![
        -1000.0_f32,
        -100.0,
        -90.0,
        -20.0,
        -1.0,
        0.0,
        1.0,
        20.0,
        100.0,
        1000.0,
    ];
    let input = T::from_data(
        TensorData::new(values.clone(), [1, values.len()]),
        (&device, DType::F32),
    )
    .require_grad();
    let output = silu(input.clone());
    let actual = output.clone().into_data().try_to_vec::<f32>().unwrap();
    let gs = output.sum().backward();
    let gradients = input
        .grad(&gs)
        .unwrap()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for ((value, result), gradient) in values.iter().zip(actual).zip(gradients) {
        let x = f64::from(*value);
        let sigmoid = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        };
        let expected = x * sigmoid;
        let derivative = sigmoid + x * sigmoid * (1.0 - sigmoid);
        assert!(
            result.is_finite() && gradient.is_finite(),
            "input {x}: result {result}, gradient {gradient}"
        );
        assert!((f64::from(result) - expected).abs() < 2e-5);
        assert!((f64::from(gradient) - derivative).abs() < 2e-5);
    }
}

#[test]
fn silu_large_magnitude_gradients_match_stable_f64_oracle() {
    verify(Device::flex().autodiff());
}

#[test]
fn former_exp_division_expression_has_nonfinite_gradients_for_finite_input() {
    let device = Device::flex().autodiff();
    let input = T::from_data([[-100.0_f32]], (&device, DType::F32)).require_grad();
    let old: T = input.clone() / (1.0 + (-input.clone()).exp());
    assert!(old.clone().into_data().try_to_vec::<f32>().unwrap()[0].is_finite());
    let gs = old.sum().backward();
    assert!(
        !input
            .grad(&gs)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0]
            .is_finite()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a visible NVIDIA GPU; run the opt-in local regression suite"]
fn cuda_silu_large_magnitude_gradients() {
    verify(Device::cuda(0).autodiff());
}
