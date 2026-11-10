//! `stateright check MODEL.json`: checks a declarative JSON model.

use stateright::declarative::{CheckOptions, JsonModel, check};
use std::process::ExitCode;

const USAGE: &str = "\
usage: stateright check [--threads N] [--max-depth N] [--output FILE] MODEL.json
       stateright --version

Checks every state of MODEL.json reachable within its bound, breadth first,
and prints a JSON report. Exit status: 0 when every invariant holds, 1 when
one is violated, 2 on a usage or model error.

  --threads N     checker threads (default: every core). Use 1 for traces
                  that are shortest paths.
  --max-depth N   explore at most N actions from an initial state, overriding
                  the model's max_depth.
  --output FILE   write the report to FILE instead of standard output.";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("stateright: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<String>) -> Result<ExitCode, String> {
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("check") => {}
        Some("--version") | Some("-V") => {
            println!("stateright {}", env!("CARGO_PKG_VERSION"));
            return Ok(ExitCode::SUCCESS);
        }
        Some("--help") | Some("-h") | Some("help") => {
            println!("{USAGE}");
            return Ok(ExitCode::SUCCESS);
        }
        _ => return Err(format!("expected a command\n{USAGE}")),
    }
    let mut options = CheckOptions::default();
    let mut output = None;
    let mut path = None;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--threads" => {
                options.threads = value("--threads")?
                    .parse()
                    .map_err(|_| "--threads takes a positive integer".to_string())?;
            }
            "--max-depth" => {
                options.max_depth = Some(
                    value("--max-depth")?
                        .parse()
                        .map_err(|_| "--max-depth takes a non-negative integer".to_string())?,
                );
            }
            "--output" => output = Some(value("--output")?),
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            a if a.starts_with('-') => return Err(format!("unknown option {a}\n{USAGE}")),
            a => {
                if path.replace(a.to_string()).is_some() {
                    return Err(format!("expected one model file\n{USAGE}"));
                }
            }
        }
    }
    let path = path.ok_or_else(|| format!("expected a model file\n{USAGE}"))?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let model = JsonModel::from_json_str(&text).map_err(|e| format!("{path}: {e}"))?;

    let report = check(model, &options).map_err(|e| format!("{path}: model error: {e}"))?;
    let text = serde_json::to_string_pretty(&report.json).unwrap();
    match output {
        Some(file) => std::fs::write(&file, text + "\n").map_err(|e| format!("{file}: {e}"))?,
        None => println!("{text}"),
    }
    let j = &report.json;
    eprintln!(
        "stateright: {} {}: {} states, {} s",
        j["model"].as_str().unwrap_or_default(),
        j["result"].as_str().unwrap_or_default(),
        j["states"],
        j["seconds"]
    );
    Ok(if report.ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
