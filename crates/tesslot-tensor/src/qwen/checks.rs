//! Device-independent decoding of finite-value reductions, shared by training guards.
use super::*;
// CubeCL returns Bool(U8/U32), while Flex uses native bool storage. Decode the
// logical value explicitly; a readback/conversion failure is not a nonfinite value.
pub(super) fn finite(tensor: &T) -> Result<bool, String> {
    finite_flag(tensor.clone().is_finite().all().into_data())
}
fn finite_flag(data: TensorData) -> Result<bool, String> {
    if !data.dtype().is_bool() {
        return Err(format!(
            "finite-check result must be Boolean, received {:?}",
            data.dtype()
        ));
    }
    let flags = data
        .try_to_vec_as::<bool>()
        .map_err(|e| format!("finite-check readback failed for {:?}: {e}", data.dtype()))?;
    match flags.as_slice() {
        [flag] => Ok(*flag),
        _ => Err(format!(
            "finite-check result must contain one Boolean, received {}",
            flags.len()
        )),
    }
}

#[cfg(test)]
mod finiteness {
    use super::*;
    use burn_tensor::BoolStore;
    #[test]
    fn decodes_native_and_integer_backed_boolean_flags() {
        for store in [BoolStore::Native, BoolStore::U8, BoolStore::U32] {
            for value in [true, false] {
                let data = TensorData::new(vec![value], [1]).convert_dtype(DType::Bool(store));
                if store != BoolStore::Native {
                    assert!(
                        data.try_to_vec::<bool>().is_err(),
                        "original strict read must reproduce the CUDA dtype mismatch"
                    );
                }
                assert_eq!(finite_flag(data).unwrap(), value, "{store:?}");
            }
        }
    }
    #[test]
    fn invalid_finite_flags_are_errors_instead_of_nonfinite_reports() {
        assert!(
            finite_flag(TensorData::new(vec![1.0f32], [1]))
                .unwrap_err()
                .contains("must be Boolean")
        );
        assert!(
            finite_flag(TensorData::new(vec![true, false], [2]))
                .unwrap_err()
                .contains("one Boolean")
        );
    }
    fn verify(device: Device) {
        for (values, expected) in [
            (vec![-1.0, 0.0, f32::MAX], true),
            (vec![1.0, f32::NAN, 0.0], false),
            (vec![f32::INFINITY, 0.0, 1.0], false),
            (vec![1.0, 0.0, f32::NEG_INFINITY], false),
        ] {
            let tensor = T::from_data(TensorData::new(values, [1, 3]), (&device, DType::F32));
            assert_eq!(finite(&tensor).unwrap(), expected);
        }
    }
    #[test]
    fn finite_guard_accepts_real_values_and_rejects_nan_and_both_infinities() {
        verify(Device::flex());
    }
    #[test]
    fn rejected_gradient_does_not_commit_weights_moments_or_step() {
        let mut model = Model::remix_demo(
            serde_json::from_slice(include_bytes!(
                "../../../../fixtures/tokenizer-identity.json"
            ))
            .unwrap(),
            7,
            42,
            true,
        )
        .unwrap();
        model.enable_adamw(Default::default(), 0.001).unwrap();
        let mut gradients = model
            .parameters
            .iter()
            .map(|(n, p)| (n.clone(), p.clone().zeros_like().detach()))
            .collect::<BTreeMap<_, _>>();
        let name = "kidra.exchange.identities.weight";
        gradients.insert(name.into(), gradients[name].clone() + f32::NAN);
        let before = model.parameters[name]
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            model
                .apply_gradients(1.0, gradients)
                .unwrap_err()
                .contains("nonfinite gradient")
        );
        assert_eq!(model.step, 0);
        assert_eq!(model.adam.as_ref().unwrap().updates, 0);
        assert!(model.adam.as_ref().unwrap().moments.is_empty());
        assert_eq!(
            model.parameters[name]
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            before
        );
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a visible NVIDIA GPU; scripts/test-cuda.sh performs hardware discovery"]
    fn cuda_finite_guard_accepts_real_values_and_rejects_nan_and_both_infinities() {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index", "--format=csv,noheader"])
            .output();
        if !output.is_ok_and(|o| o.status.success() && !o.stdout.is_empty())
            || std::env::var("CUDA_VISIBLE_DEVICES")
                .is_ok_and(|v| v.trim().is_empty() || v.trim() == "-1")
        {
            eprintln!("SKIP CUDA finite guard: NVIDIA discovery unavailable");
            return;
        }
        verify(Device::cuda(0));
    }
}
