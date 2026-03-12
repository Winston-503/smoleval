use std::io::Write;

use smoleval::EvalReport;

/// Write a human-readable text report.
pub fn format_text(report: &EvalReport, threshold: f64, w: &mut dyn Write) -> std::io::Result<()> {
    writeln!(w, "=== {} ===\n", report.dataset_name())?;

    for result in report.results() {
        if let Some(err) = result.outcome().error() {
            writeln!(
                w,
                "[ERROR] {} [{:.1}s]",
                result.test_case().name(),
                result.duration().as_secs_f64()
            )?;
            writeln!(w, "  {err}")?;
        } else {
            writeln!(
                w,
                "[{}] {} ({:.2}) [{:.1}s]",
                result.label(),
                result.test_case().name(),
                result.score(),
                result.duration().as_secs_f64()
            )?;

            for (def, check_result) in result.test_case().checks().iter().zip(result.check_results()) {
                writeln!(
                    w,
                    "  [{}] {}: {}",
                    check_result.label(),
                    def.kind(),
                    check_result.reason()
                )?;
            }
        }
        writeln!(w)?;
    }

    let errored = report.errored_count();
    let errored_str = if errored > 0 {
        format!(" ({errored} errored)")
    } else {
        String::new()
    };

    let threshold_str = if threshold < 1.0 {
        format!(" | Threshold: {threshold:.2}")
    } else {
        String::new()
    };

    writeln!(
        w,
        "Results: {}/{} passed{} | Mean score: {:.2}{} | Time: {:.1}s",
        report.passed_count(),
        report.total_count(),
        errored_str,
        report.mean_score(),
        threshold_str,
        report.duration().as_secs_f64()
    )?;

    Ok(())
}

/// Write a JSON report for programmatic consumption.
pub fn format_json(report: &EvalReport, threshold: f64, w: &mut dyn Write) -> std::io::Result<()> {
    let json = serde_json::json!({
        "datasetName": report.dataset_name(),
        "summary": {
            "totalCount": report.total_count(),
            "passedCount": report.passed_count(),
            "failedCount": report.failed_count(),
            "erroredCount": report.errored_count(),
            "meanScore": report.mean_score(),
            "threshold": threshold,
            "durationSecs": report.duration().as_secs_f64(),
        },
        "results": report.results().iter().map(|r| {
            let mut obj = serde_json::json!({
                "testCase": r.test_case().name(),
                "score": r.score(),
                "passed": r.score() == 1.0,
                "durationSecs": r.duration().as_secs_f64(),
                "checks": r.test_case().checks().iter().zip(r.check_results()).map(|(def, cr)| {
                    serde_json::json!({
                        "kind": def.kind(),
                        "score": cr.score(),
                        "passed": cr.passed(),
                        "reason": cr.reason(),
                    })
                }).collect::<Vec<_>>(),
            });
            if let Some(err) = r.outcome().error() {
                obj["error"] = serde_json::json!(err);
            }
            obj
        }).collect::<Vec<_>>(),
    });

    writeln!(w, "{}", serde_json::to_string_pretty(&json).unwrap())?;
    Ok(())
}

/// Write a JUnit XML report.
pub fn format_junit(report: &EvalReport, w: &mut dyn Write) -> std::io::Result<()> {
    let failures = report
        .results()
        .iter()
        .filter(|r| !r.outcome().is_error() && r.score() < 1.0)
        .count();
    let errors = report.errored_count();

    writeln!(w, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        w,
        r#"<testsuites tests="{}" failures="{}" errors="{}" time="{:.3}">"#,
        report.total_count(),
        failures,
        errors,
        report.duration().as_secs_f64()
    )?;
    writeln!(
        w,
        r#"  <testsuite name="{}" tests="{}" failures="{}" errors="{}" time="{:.3}">"#,
        xml_escape(report.dataset_name()),
        report.total_count(),
        failures,
        errors,
        report.duration().as_secs_f64()
    )?;

    for result in report.results() {
        write!(
            w,
            r#"    <testcase name="{}" time="{:.3}""#,
            xml_escape(result.test_case().name()),
            result.duration().as_secs_f64()
        )?;

        if let Some(err) = result.outcome().error() {
            writeln!(w, ">")?;
            writeln!(w, r#"      <error message="{}" />"#, xml_escape(err))?;
            writeln!(w, "    </testcase>")?;
        } else if result.score() < 1.0 {
            writeln!(w, ">")?;
            // Collect failed check details for the failure body
            let mut details = String::new();
            for (def, cr) in result.test_case().checks().iter().zip(result.check_results()) {
                if !cr.passed() {
                    if !details.is_empty() {
                        details.push('\n');
                    }
                    details.push_str(&format!("[{}] {}: {}", cr.label(), def.kind(), cr.reason()));
                }
            }
            writeln!(
                w,
                r#"      <failure message="score: {:.2}">{}</failure>"#,
                result.score(),
                xml_escape(&details)
            )?;
            writeln!(w, "    </testcase>")?;
        } else {
            writeln!(w, " />")?;
        }
    }

    writeln!(w, "  </testsuite>")?;
    writeln!(w, "</testsuites>")?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
