use thiserror::Error;

#[derive(Debug, Error)]
pub enum SmolError {
    #[error("failed to read dataset: {0}")]
    DatasetIo(#[from] std::io::Error),

    #[error("failed to parse YAML: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    #[error("agent error: {0}")]
    AgentError(String),

    #[error("invalid check score {0}: must be between 0.0 and 1.0")]
    InvalidScore(f64),

    #[error("unknown check type: {0}")]
    UnknownCheck(String),

    #[error("invalid check config: {0}")]
    CheckConfig(String),
}

pub type Result<T> = std::result::Result<T, SmolError>;
