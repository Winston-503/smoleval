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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_io_error_display() {
        let err = SmolError::DatasetIo(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file missing",
        ));
        let msg = err.to_string();
        assert!(msg.contains("failed to read dataset"));
        assert!(msg.contains("file missing"));
    }

    #[test]
    fn agent_error_display() {
        let err = SmolError::AgentError("timeout".into());
        assert_eq!(err.to_string(), "agent error: timeout");
    }

    #[test]
    fn invalid_score_display() {
        let err = SmolError::InvalidScore(2.5);
        let msg = err.to_string();
        assert!(msg.contains("2.5"));
        assert!(msg.contains("between 0.0 and 1.0"));
    }

    #[test]
    fn unknown_check_display() {
        let err = SmolError::UnknownCheck("fooCheck".into());
        assert_eq!(err.to_string(), "unknown check type: fooCheck");
    }

    #[test]
    fn check_config_display() {
        let err = SmolError::CheckConfig("missing field".into());
        assert_eq!(err.to_string(), "invalid check config: missing field");
    }

    #[test]
    fn io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err: SmolError = io_err.into();
        assert!(matches!(err, SmolError::DatasetIo(_)));
    }

    #[test]
    fn yaml_error_converts() {
        let yaml_err = serde_yaml::from_str::<serde_yaml::Value>("{{bad").unwrap_err();
        let err: SmolError = yaml_err.into();
        assert!(matches!(err, SmolError::YamlParse(_)));
    }
}
