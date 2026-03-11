use smoleval::EvalReport;

/// Print a human-readable text report.
pub fn print_text(report: &EvalReport) {
    println!("=== {} ===\n", report.dataset_name);

    for result in &report.results {
        let status = if result.score == 1.0 { "PASS" } else { "FAIL" };
        println!("[{status}] {} ({:.2})", result.test_case.name, result.score);

        for (def, check_result) in result.test_case.checks.iter().zip(&result.check_results) {
            let icon = if check_result.passed() { "OK" } else { "FAIL" };
            println!("  [{icon}] {}: {}", def.check_type, check_result.reason());
        }
        println!();
    }

    println!(
        "Results: {}/{} passed | Mean score: {:.2}",
        report.passed_count(),
        report.total_count(),
        report.mean_score()
    );
}

/// Print a JSON report for programmatic consumption.
pub fn print_json(report: &EvalReport) -> anyhow::Result<()> {
    // Build a serializable representation
    let json = serde_json::json!({
        "datasetName": report.dataset_name,
        "summary": {
            "totalCount": report.total_count(),
            "passedCount": report.passed_count(),
            "failedCount": report.failed_count(),
            "meanScore": report.mean_score(),
        },
        "results": report.results.iter().map(|r| {
            serde_json::json!({
                "testCase": r.test_case.name,
                "score": r.score,
                "passed": r.score == 1.0,
                "checks": r.test_case.checks.iter().zip(&r.check_results).map(|(def, cr)| {
                    serde_json::json!({
                        "type": def.check_type,
                        "score": cr.score(),
                        "passed": cr.passed(),
                        "reason": cr.reason(),
                    })
                }).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
