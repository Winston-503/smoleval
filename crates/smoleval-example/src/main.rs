use smoleval::{Agent, ExactMatch, Sample, evaluate};

/// A toy agent that uppercases its input.
struct UpperAgent;

impl Agent for UpperAgent {
    fn run(&self, input: &str) -> String {
        input.to_uppercase()
    }
}

fn main() {
    let samples = vec![
        Sample { input: "hello".into(), expected: "HELLO".into() },
        Sample { input: "world".into(), expected: "WORLD".into() },
        Sample { input: "Rust".into(), expected: "RUST".into() },
        Sample { input: "oops".into(), expected: "nope".into() },
    ];

    let report = evaluate(&UpperAgent, &ExactMatch, &samples);
    println!("{report}");
}
