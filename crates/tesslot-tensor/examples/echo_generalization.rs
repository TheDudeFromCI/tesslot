//! Opt-in multi-pattern training/evaluation; no downloads or GPU dependency.
use tesslot_tensor::qwen::{Model, corpus::Split};
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let steps = args
        .first()
        .map(|v| {
            v.parse::<usize>()
                .map_err(|_| "expected an integer step count".to_string())
        })
        .transpose()?
        .unwrap_or(600);
    let optimizer = args.get(1).map(String::as_str).unwrap_or("adamw");
    if args.len() > 2 || !(1..=4000).contains(&steps) || !matches!(optimizer, "adamw" | "sgd") {
        return Err("use: echo_generalization [1..4000 updates] [adamw|sgd]".into());
    }
    let mut model = Model::echo_demo(true)?;
    model.start_echo_corpus(Default::default())?;
    if optimizer == "adamw" {
        model.enable_adamw(Default::default(), 0.003)?;
    }
    let before = model.evaluate_echo_corpus(Split::Development, 16)?;
    println!(
        "optimizer={} initial development: loss={} free_running_accuracy={}",
        model.optimizer_id(),
        before.loss,
        before.free_running_accuracy
    );
    for _ in 0..steps {
        let metric = model.train_echo_corpus_step()?;
        if metric.step.is_multiple_of(100) {
            println!(
                "step={} train_loss={} gradient_norm={}",
                metric.step, metric.loss, metric.gradient_norm
            );
        }
    }
    // SGD is a development comparison; the declared AdamW recipe runs the final test.
    let splits = if optimizer == "adamw" {
        vec![Split::Development, Split::Test]
    } else {
        vec![Split::Development]
    };
    for split in splits {
        println!(
            "{}",
            serde_json::to_string_pretty(&model.evaluate_echo_corpus(split, 16)?)
                .map_err(|e| e.to_string())?
        );
    }
    Ok(())
}
