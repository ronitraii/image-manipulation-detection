# Contributing to Image Manipulation Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone <repo-url>
cd "AI BAsed Forgery Detection"

# Install frontend dependencies
cd apps/frontend
npm install
cd ../..

# Setup Python backend
cd apps/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Development Servers

**Option 1: Simultaneous with Docker**
```bash
docker-compose up
```

**Option 2: Individual Services**
```bash
# Terminal 1 - Frontend
cd apps/frontend
npm run dev

# Terminal 2 - Backend
cd apps/backend
python app.py
```

## Commit Conventions

We follow conventional commits for clear commit history:

- `feat:` - New features
- `fix:` - Bug fixes
- `build:` - Build system changes
- `ci:` - CI/CD configuration
- `docs:` - Documentation updates
- `refactor:` - Code refactoring
- `test:` - Test updates

Example: `feat: Add segmentation model improvements`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commits
3. Test thoroughly before submitting
4. Push to your fork and create a pull request
5. Ensure all CI checks pass

## Code Style

### Frontend (TypeScript/React)
- Use ESLint configuration
- Format with Prettier
- Follow TypeScript best practices
- Use functional components with hooks

### Backend (Python)
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for functions
- Keep functions focused and modular

## Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, versions)
- Screenshots/logs if applicable

## License

By contributing, you agree that your contributions will be licensed under the same MIT license as the project.
