# Image Manipulation Detection Platform

A full-stack application for detecting image manipulation using deep learning models. This monorepo contains both the frontend (Next.js) and backend (Flask) components.

## Project Structure

```
├── apps/
│   ├── frontend/     # Next.js React frontend
│   └── backend/      # Flask Python backend
├── package.json      # Root workspace configuration
└── README.md
```

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "AI BAsed Forgery Detection"

# Install frontend dependencies
npm install --workspace=frontend

# Install backend dependencies (create virtual environment first)
cd apps/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install flask pyngrok torch torchvision segmentation-models-pytorch pillow flask-cors
```

### Running Both Services

```bash
# From root directory - runs both frontend and backend
npm run dev
```

### Running Individually

```bash
# Frontend only (Next.js on http://localhost:3000)
npm run dev --workspace=frontend

# Backend only (Flask on http://localhost:5000)
cd apps/backend
python app.py
```

## Features

- **Frontend**: Modern Next.js/React UI with Tailwind CSS
- **Backend**: Flask API with DeepLabV3+ segmentation and ResNet50 classification
- **ML Models**: Deep learning for image manipulation detection
  - Segmentation: Pixel-level localization of manipulated regions
  - Classification: Identifies manipulation types (Splicing, Copy-Move, Inpainting, Face Manipulation)

## Documentation

- [Frontend README](./apps/frontend/README.md)
- [Backend README](./apps/backend/README.md)

## License

MIT
