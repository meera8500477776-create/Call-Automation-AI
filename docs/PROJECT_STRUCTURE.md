# Project Structure

## Directory Layout

```
config-driven-voice-agent/
├── src/
│   └── voice_agent/              # Main application package
│       ├── __init__.py
│       ├── core/                 # Core orchestration components
│       │   └── __init__.py
│       ├── services/             # External service integrations
│       │   └── __init__.py       # STT, LLM, TTS services
│       ├── adapters/             # Industry-specific adapters
│       │   └── __init__.py       # FHIR (healthcare), DMS (automotive)
│       ├── security/             # Security and encryption
│       │   └── __init__.py       # Authentication, encryption, audit
│       ├── config/               # Configuration management
│       │   ├── __init__.py
│       │   ├── settings.py       # Environment settings
│       │   └── loader.py         # YAML/TOML config loader
│       ├── monitoring/           # Observability and logging
│       │   ├── __init__.py
│       │   └── logger.py         # Structured logging
│       └── utils/                # Utility functions
│           └── __init__.py
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── config/                  # Configuration tests
│   │   ├── __init__.py
│   │   ├── test_loader.py
│   │   └── test_settings.py
│   └── monitoring/              # Monitoring tests
│       ├── __init__.py
│       └── test_logger.py
├── config/                       # Tenant configurations
│   └── tenants/
│       ├── example_healthcare.yaml
│       └── example_automotive.yaml
├── docs/                         # Documentation
│   └── PROJECT_STRUCTURE.md
├── data/                         # Runtime data (created at runtime)
├── logs/                         # Log files (created at runtime)
├── certs/                        # TLS certificates (user-provided)
├── .env.example                  # Environment template
├── .env                          # Environment variables (git-ignored)
├── .gitignore                    # Git ignore rules
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependency list
├── setup.sh                     # Setup script
├── README.md                    # Project overview
└── SETUP_NOTES.md              # Setup instructions
```

## Module Descriptions

### Core (`src/voice_agent/core/`)

Contains the main orchestration components:
- **Pipecat Orchestrator**: Manages the voice pipeline lifecycle
- **Session Manager**: Handles conversation state and context
- **Voice Pipeline**: Coordinates STT, LLM, TTS, and VAD components

### Services (`src/voice_agent/services/`)

External service integrations:
- **STT Service**: Deepgram integration for speech-to-text
- **LLM Service**: OpenAI/Ollama/vLLM integration for language understanding
- **TTS Service**: ElevenLabs/OpenAI integration for text-to-speech
- **VAD Module**: Silero VAD for voice activity detection

### Adapters (`src/voice_agent/adapters/`)

Industry-specific integrations:
- **FHIR Adapter**: Healthcare EHR integration via SMART-on-FHIR
- **DMS Adapter**: Automotive dealership management system integration

### Security (`src/voice_agent/security/`)

Security and compliance features:
- **Encryption Service**: AES-256 encryption and TLS management
- **Authentication Service**: OAuth 2.0 client for API authentication
- **Audit Logger**: Immutable audit logging for compliance

### Config (`src/voice_agent/config/`)

Configuration management:
- **Settings**: Environment variable configuration using Pydantic
- **Loader**: YAML/TOML configuration file loader
- **Tenant Registry**: Multi-tenant configuration management

### Monitoring (`src/voice_agent/monitoring/`)

Observability and monitoring:
- **Logger**: Structured logging using structlog
- **Latency Monitor**: Per-stage latency tracking
- **Observability Platform**: Integration with LangSmith/Helicone/Twilio CI

### Utils (`src/voice_agent/utils/`)

Utility functions and helpers for common operations.

## Configuration Files

### `pyproject.toml`

Project metadata and dependencies using modern Python packaging standards.

### `.env` / `.env.example`

Environment variables for API keys, configuration paths, and runtime settings.

### `config/tenants/*.yaml`

Tenant-specific configurations defining:
- Persona and voice settings
- LLM provider and model
- Tool definitions and API endpoints
- Knowledge base references
- Constraints and policies

## Testing Structure

Tests mirror the source structure:
- Unit tests for individual components
- Property-based tests for correctness properties
- Integration tests for component interactions
- End-to-end tests for complete workflows

## Data Directories

### `data/`

Runtime data storage (created automatically):
- Session database
- Cached configurations
- Temporary files

### `logs/`

Application logs (created automatically):
- Structured JSON logs
- Audit logs
- Error logs

### `certs/`

TLS certificates (user-provided):
- `cert.pem`: TLS certificate
- `key.pem`: Private key

## Development Workflow

1. **Setup**: Run `./setup.sh` or follow manual setup in SETUP_NOTES.md
2. **Configure**: Edit `.env` with API keys
3. **Develop**: Add code to appropriate module
4. **Test**: Write tests in corresponding test directory
5. **Validate**: Run `pytest`, `black`, `ruff`, `mypy`
6. **Deploy**: Use Docker/Kubernetes configurations (to be added)

## Next Steps

As development progresses, additional directories will be added:
- `docker/`: Docker and docker-compose configurations
- `k8s/`: Kubernetes manifests
- `scripts/`: Utility scripts for deployment and maintenance
- `examples/`: Example usage and integration code
