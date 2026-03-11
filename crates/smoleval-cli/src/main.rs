mod report;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use smoleval::{CheckRegistry, EvalDataset, evaluate};

#[derive(Parser)]
#[command(
    name = "smoleval",
    about = "Minimal AI agent evaluation framework",
    version
)]
struct Cli {
    /// Path to the YAML evaluation dataset.
    #[arg(short, long)]
    dataset: PathBuf,

    /// Agent endpoint URL.
    #[arg(short, long)]
    agent: String,

    /// Output format.
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Request timeout in seconds.
    #[arg(short, long, default_value = "60")]
    timeout: u64,
}

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load dataset
    let dataset = EvalDataset::from_file(&cli.dataset)?;

    // Create check registry with built-in checks
    let registry = CheckRegistry::with_builtins();

    let timeout = Duration::from_secs(cli.timeout);
    let agent = smoleval_http::HttpAgent::with_timeout(&cli.agent, timeout);
    let report = evaluate(&agent, &dataset, &registry).await?;

    // Output report
    match cli.format {
        OutputFormat::Text => report::print_text(&report),
        OutputFormat::Json => report::print_json(&report)?,
    }

    // Exit with non-zero code if any test failed
    if report.failed_count() > 0 {
        std::process::exit(1);
    }

    Ok(())
}
