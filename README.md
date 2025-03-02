# ICD-10-CM Coding Assistant

An intelligent medical coding assistant that helps healthcare professionals find and validate ICD-10-CM codes efficiently.

## ✨ Features

- 🔍 **Quick Code Lookup**: 
  - Instant search with direct and similar matches
  - Collapsible results with codes and modifiers
  - Support for partial term matching

- 📚 **Browse by Letter**: 
  - Alphabetical navigation of medical terms
  - Organized display of codes by category
  - Quick access to common conditions

- 📝 **Clinical Note Analysis**: (Under Development)
  - Natural language processing of clinical notes
  - Automated code suggestions
  - Context-aware term extraction

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.12
- **Database**: Neo4j Graph Database
- **APIs**: OpenAI GPT (for clinical analysis)

## 📦 Prerequisites

- Python 3.12+
- Docker (for Neo4j)
- Git

## 🚀 Quick Start

1. **Clone & Setup**
```bash
git clone https://github.com/amit-chaubey/icd10-cm-project.git
cd icd10-cm-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Neo4j Database**
```bash
# Start Neo4j container
docker run \
  --name neo4j-icd \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS=["apoc"] \
  neo4j:latest
```

4. **Configure Environment**
```bash
cp .env.example .env

# Edit .env with your credentials:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password
# OPENAI_API_KEY=your_api_key  # Optional, for clinical analysis
```

5. **Launch Application**
```bash
streamlit run frontend/app.py
```

## 💡 Usage Examples

### Quick Code Lookup
```text
Search: "Type 2 diabetes"
Results:
- E11.9 (Type 2 diabetes mellitus without complications)
- E11.8 (Type 2 diabetes mellitus with other specified complications)
```

### Browse by Letter
```text
Letter: "A"
Results:
- Asthma (J45.909)
- Acute bronchitis (J20.9)
- Anxiety disorder (F41.9)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_neo4j_client.py

# Run with coverage
pytest --cov=backend
```

## 🔧 Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
flake8 .

# Format code
black .
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 🚀 Roadmap

- [ ] Enhanced Clinical Note Analysis
- [ ] Bulk Code Processing
- [ ] Export to CSV/Excel
- [ ] User Authentication
- [ ] Advanced Search Filters
- [ ] Integration with EHR Systems
