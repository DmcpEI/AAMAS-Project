# Hunter-Prey Nash Q-Learning Simulation

A sophisticated multi-agent simulation featuring hunters and prey agents with Nash Q-Learning capabilities, built using the Mesa framework and providing real-time visualization through Solara interface.

## 🎯 Project Overview

This project implements a competitive multi-agent environment where various AI strategies compete in a hunter-prey scenario. The simulation showcases:

- **Nash Q-Learning**: Advanced reinforcement learning with game-theoretic optimization
- **Minimax Algorithm**: Tree search with alpha-beta pruning for strategic gameplay  
- **Real-time Visualization**: Interactive web interface with live Q-table monitoring
- **Multiple Agent Types**: Random, greedy, and AI-powered hunters and prey

## 🛠️ Prerequisites

- **WINDOWS ONLY**
- **Python 3.7+**
- **pip** package manager

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AAMAS-Project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Windows only
pip install mesa solara altair networkx matplotlib
```

## 🚀 Running the Simulation

### Basic Execution
```bash
cd src
solara run ./run.py
```

### Access the Web Interface
1. After running the command, you'll see output similar to:
   ```
   Solara server starting...
   * Running on http://localhost:8765
   ```
2. Open your web browser and navigate to the provided URL (typically `http://localhost:8765`)

## 🎮 Using the Simulation

### Web Interface Components

1. **Grid Visualization**: Live view of agent positions and movements
2. **Control Panel**: Adjust simulation parameters using sliders
3. **Performance Charts**: Real-time metrics and statistics
4. **Q-Table Display**: Live Q-table analysis for learning agents
5. **Kill Notifications**: Real-time alerts when hunters catch prey
6. **Status Dashboard**: Current simulation state and agent counts

### Agent Types and Visual Representation

| Agent Type | Color | Shape | Description |
|------------|-------|-------|-------------|
| RandomHunter | Yellow | Circle | Random movement hunter |
| GreedyHunter | Light Yellow | Circle | Greedy pursuit hunter |
| NashQHunter | Red | Large Circle | Nash Q-Learning hunter |
| MinimaxHunter | Dark Red | Square | Minimax algorithm hunter |
| Prey | Blue | Circle | Basic prey with random movement |
| NashQPrey | Green | Large Circle | Nash Q-Learning prey |
| MinimaxPrey | Dark Green | Square | Minimax algorithm prey |

### Configuration Parameters

Use the web interface sliders to adjust:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Number of Hunters | 0-10 | 0 | Random movement hunters |
| Number of Greedy Hunters | 0-10 | 0 | Greedy pursuit hunters |
| Number of Nash Q-learning Hunters | 0-10 | 1 | AI learning hunters |
| Number of Minimax Hunters | 0-10 | 0 | Strategic tree-search hunters |
| Number of Preys | 0-20 | 0 | Basic random prey |
| Number of Nash Q-learning Preys | 0-20 | 1 | AI learning prey |
| Number of Minimax Preys | 0-20 | 0 | Strategic tree-search prey |
| Minimax Search Depth | 1-6 | 4 | Depth of minimax tree search |
| Grid Width | 1-20 | 4 | Environment width |
| Grid Height | 1-20 | 4 | Environment height |

## 🧪 Testing and Experimentation

### Basic Testing Scenarios

#### 1. Simple Hunter-Prey Scenario
```
- Nash Q-learning Hunters: 1
- Nash Q-learning Preys: 2
- Grid: 4x4
```

#### 2. Multi-Algorithm Comparison
```
- Greedy Hunters: 1
- Nash Q-learning Hunters: 1
- Minimax Hunters: 1
- Nash Q-learning Preys: 3
- Grid: 6x6
```

#### 3. Large-Scale Environment
```
- Various hunter types: 2-3 each
- Nash Q-learning Preys: 5-8
- Grid: 10x10+
```

### Monitoring Learning Progress

1. **Q-Table Analysis**: Watch the Q-table values evolve in real-time
2. **Performance Metrics**: Monitor kill rates, rewards, and survival statistics
3. **Terminal Output**: Detailed logging of agent decisions and learning updates

### Performance Considerations

- **Minimax Search Depth**: Higher depth = better strategy but slower computation
- **Grid Size**: Larger grids = more complex behaviors but less frequent interactions
- **Agent Population**: More agents = richer interactions but potentially slower simulation

## 🔧 Advanced Configuration

### Logging Configuration
The simulation includes comprehensive logging. To modify logging levels, edit the `setup_logging()` function in `run.py`:

```python
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed output
    force=True,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)
```

### Learning Parameters
Nash Q-Learning parameters are configured in the agent classes:
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.9  
- **Exploration Rate (ε)**: 0.3-0.9
- **Epsilon Decay**: 0.995-0.998

### Reward Structure
- **Hunter Successful Kill**: +10.0
- **Hunter Movement Cost**: -0.1
- **Prey Survival**: +0.1
- **Prey Death Penalty**: -10.0

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing process on port 8765
   # Windows:
   netstat -ano | findstr :8765
   taskkill /PID <PID> /F
   
   # Linux/Mac:
   lsof -ti:8765 | xargs kill -9
   ```

2. **Import Errors**
   - Ensure you're in the `src` directory when running
   - Verify all dependencies are installed
   - Check virtual environment is activated

3. **Slow Performance**
   - Reduce minimax search depth
   - Decrease number of agents
   - Use smaller grid sizes

### Virtual Environment Issues
```bash
# Recreate virtual environment if needed
deactivate
rmdir /s venv  # Windows

python -m venv venv
venv\Scripts\activate  # Windows
pip install mesa solara altair networkx matplotlib
```

## 📁 Project Structure

```
AAMAS-Project/
├── README.md                    # This file
├── PROJECT_DOCUMENTATION.md     # Detailed technical documentation
└── src/                         # Source code directory
    ├── run.py                   # Main application entry point
    ├── ChartMetrics.py          # Metrics management
    ├── QTableDisplayer.py       # Q-table analysis and display
    ├── Agents/                  # Agent implementations
    │   ├── BaseAgent.py         # Base agent functionality
    │   ├── NashQAgent.py        # Nash Q-Learning base class
    │   ├── MinimaxAgent.py      # Minimax algorithm base class
    │   ├── NashSolver.py        # Nash equilibrium computation
    │   ├── Hunters/             # Hunter agent types
    │   └── Preys/               # Prey agent types
    └── Models/
        └── HunterPreyModel.py   # Main simulation model
```

## 🎓 Educational Applications

This simulation is ideal for:
- **Multi-Agent Systems Research**
- **Game Theory Education**
- **Reinforcement Learning Studies**
- **Algorithm Comparison Analysis**
- **AI Strategy Development**

## 📚 Further Reading

- See `PROJECT_DOCUMENTATION.md` for detailed technical information
- Mesa Framework: https://mesa.readthedocs.io/
- Solara Documentation: https://solara.dev/
- Nash Q-Learning: Research papers on multi-agent reinforcement learning

## 👥 Authors

**AASMA Project Team**

Guilherme Freitas | Diogo Paixão | Gonçalo Silva

---
