mod report;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use smoleval::{CheckRegistry, EvalDataset, EvalOptions, evaluate_with_options};

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Junit,
}

#[derive(Parser)]
#[command(name = "smoleval", about = "Minimal AI agent evaluation framework", version)]
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

    /// Minimum mean score to pass (0.0 to 1.0).
    #[arg(long, default_value = "1.0")]
    threshold: f64,

    /// Write report to a file (in addition to stdout, unless --quiet).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Suppress stdout output.
    #[arg(short, long)]
    quiet: bool,

    /// Number of test cases to run concurrently.
    #[arg(long, default_value = "1")]
    concurrency: usize,

    /// Abort on first agent error (only effective with concurrency=1).
    #[arg(long)]
    fail_fast: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    anyhow::ensure!(
        (0.0..=1.0).contains(&cli.threshold),
        "threshold must be between 0.0 and 1.0"
    );

    let dataset = EvalDataset::from_file(&cli.dataset)?;
    let registry = CheckRegistry::with_builtins();

    let timeout = Duration::from_secs(cli.timeout);
    let agent = smoleval_http::HttpAgent::with_timeout(&cli.agent, timeout);

    let options = EvalOptions {
        concurrency: cli.concurrency,
        fail_fast: cli.fail_fast,
    };
    let report = evaluate_with_options(&agent, &dataset, &registry, &options).await?;

    if !cli.quiet {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        write_report(&report, &cli.format, cli.threshold, &mut out)?;
    }

    if let Some(ref path) = cli.output {
        let file = File::create(path)?;
        let mut out = BufWriter::new(file);
        write_report(&report, &cli.format, cli.threshold, &mut out)?;
    }

    if report.mean_score() < cli.threshold {
        std::process::exit(1);
    }

    Ok(())
}

fn write_report(
    report: &smoleval::EvalReport,
    format: &OutputFormat,
    threshold: f64,
    w: &mut dyn std::io::Write,
) -> Result<()> {
    match format {
        OutputFormat::Text => report::format_text(report, threshold, w)?,
        OutputFormat::Json => report::format_json(report, threshold, w)?,
        OutputFormat::Junit => report::format_junit(report, w)?,
    }
    Ok(())
}
