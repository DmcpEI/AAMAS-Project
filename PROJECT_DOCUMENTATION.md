# Hunter-Prey Nash Q-Learning Simulation Project Documentation

## Project Overview

The **Hunter-Prey Nash Q-Learning Simulation** is a comprehensive multi-agent system simulation built using the Mesa framework. This project implements various artificial intelligence algorithms in a competitive hunter-prey environment, focusing on Nash Q-Learning, Minimax algorithms, and traditional heuristic approaches.

### Key Features

- **Multi-Agent Environment**: Supports multiple types of hunters and prey agents with different AI strategies
- **Nash Q-Learning Implementation**: Advanced reinforcement learning with game-theoretic Nash equilibrium computation
- **Minimax Algorithm**: Tree search with alpha-beta pruning for strategic gameplay
- **Real-time Visualization**: Interactive web-based interface using Solara for live simulation monitoring
- **Q-Table Analysis**: Real-time Q-table display and analysis for learning agents
- **Performance Metrics**: Comprehensive tracking of agent performance, kills, rewards, and learning progress

## Project Structure

```
AAMAS-Project/
├── README.md                    # Basic project information
└── src/                         # Source code directory
    ├── run.py                   # Main application entry point
    ├── ChartMetrics.py          # Metrics management for visualization
    ├── QTableDisplayer.py       # Q-table analysis and display
    ├── Agents/                  # Agent implementations
    │   ├── BaseAgent.py         # Base agent class with shared functionality
    │   ├── NashQAgent.py        # Nash Q-Learning agent base class
    │   ├── MinimaxAgent.py      # Minimax algorithm base class
    │   ├── NashSolver.py        # Nash equilibrium computation utilities
    │   ├── Hunters/             # Hunter agent implementations
    │   │   ├── RandomHunter.py      # Random movement hunter
    │   │   ├── GreedyHunter.py      # Greedy pursuit hunter
    │   │   ├── NashQHunter.py       # Nash Q-Learning hunter
    │   │   └── MinimaxHunter.py     # Minimax algorithm hunter
    │   └── Preys/               # Prey agent implementations
    │       ├── Prey.py              # Basic prey with random movement
    │       ├── NashQPrey.py         # Nash Q-Learning prey
    │       └── MinimaxPrey.py       # Minimax algorithm prey
    └── Models/
        └── HunterPreyModel.py   # Main simulation model and coordination
```

## Agent Types and AI Algorithms

### Hunter Agents

#### 1. **RandomHunter** (`Agents/Hunters/RandomHunter.py`)
- **Strategy**: Random movement across the grid
- **Purpose**: Baseline comparison for other algorithms
- **Characteristics**: No learning, purely stochastic behavior

#### 2. **GreedyHunter** (`Agents/Hunters/GreedyHunter.py`)
- **Strategy**: Always moves toward the closest prey
- **Algorithm**: Greedy best-first search using Manhattan distance
- **Characteristics**: Deterministic, no learning, immediate gratification

#### 3. **NashQHunter** (`Agents/Hunters/NashQHunter.py`)
- **Strategy**: Nash Q-Learning with game-theoretic optimization
- **Algorithm**: Q-Learning enhanced with Nash equilibrium computation
- **Key Features**:
  - Learns optimal strategies against intelligent opponents
  - Computes Nash equilibria for action selection
  - Handles multi-agent interactions
  - Adaptive exploration with loop detection

#### 4. **MinimaxHunter** (`Agents/Hunters/MinimaxHunter.py`)
- **Strategy**: Minimax tree search with alpha-beta pruning
- **Algorithm**: Game tree search optimizing for capture probability
- **Key Features**:
  - Configurable search depth (1-6 levels)
  - Alpha-beta pruning for efficiency
  - Strategic positioning and prey cornering
  - Anticipation of prey movements

### Prey Agents

#### 1. **Prey** (`Agents/Preys/Prey.py`)
- **Strategy**: Random movement for survival
- **Purpose**: Basic prey agent for testing hunter effectiveness
- **Characteristics**: No learning, purely evasive random behavior

#### 2. **NashQPrey** (`Agents/Preys/NashQPrey.py`)
- **Strategy**: Nash Q-Learning for optimal survival
- **Algorithm**: Q-Learning with Nash equilibrium against hunters
- **Key Features**:
  - Learns to exploit hunter weaknesses
  - Cooperative/competitive behavior with other prey
  - Death penalty learning (-10 reward)
  - Respawn mechanism for continuous learning

#### 3. **MinimaxPrey** (`Agents/Preys/MinimaxPrey.py`)
- **Strategy**: Minimax tree search for evasion
- **Algorithm**: Game tree search minimizing capture probability
- **Key Features**:
  - Configurable search depth
  - Strategic positioning away from hunters
  - Mobility optimization
  - Prediction of hunter movements

## Core Algorithms and Technologies

### Nash Q-Learning Implementation

The Nash Q-Learning algorithm is implemented in `Agents/NashQAgent.py` and represents the most sophisticated AI approach in the project:

#### **Algorithm Components**:
1. **Q-Table Structure**: `(state, my_action, other_action) → value`
2. **Nash Equilibrium Computation**: Uses `NashSolver.py` for game-theoretic optimization
3. **3-Phase Synchronization**:
   - **Phase 1**: All agents choose actions simultaneously
   - **Phase 2**: Actions executed, states observed
   - **Phase 3**: Q-tables updated with synchronized experiences

#### **Key Features**:
- **Optimistic Initialization**: Encourages exploration of unknown state-action pairs
- **Loop Detection**: Prevents agents from getting stuck in repetitive patterns
- **Adaptive Exploration**: Dynamic epsilon adjustment based on learning progress
- **Multi-Agent Coordination**: Handles interactions between multiple learning agents

### Minimax Algorithm Implementation

The Minimax algorithm is implemented in `Agents/MinimaxAgent.py` with the following features:

#### **Algorithm Components**:
1. **Tree Search**: Recursive game tree exploration
2. **Alpha-Beta Pruning**: Optimization to reduce search space
3. **Evaluation Function**: Strategic position assessment
4. **Move Prediction**: Anticipation of opponent actions

#### **Strategic Elements**:
- **Hunters**: Maximize capture probability, corner prey, block escape routes
- **Prey**: Minimize capture risk, maintain mobility, strategic positioning

### Model Coordination and Synchronization

The main simulation model (`Models/HunterPreyModel.py`) coordinates all agents and implements:

#### **Synchronization Systems**:
1. **Nash Q-Learning 3-Phase System**: Ensures proper multi-agent learning
2. **Kill Detection and Handling**: Manages hunter-prey interactions
3. **Respawn Mechanisms**: Maintains population dynamics
4. **Collision Detection**: Prevents agent overlap conflicts

#### **Data Collection**:
- Real-time metrics tracking for all agent types
- Performance statistics (kills, rewards, survival rates)
- Learning progress monitoring for Q-Learning agents

## Visualization and User Interface

### Real-Time Web Interface (`run.py`)

The project provides a sophisticated web-based interface using Solara:

#### **Components**:
1. **Grid Visualization**: Live agent positions with color-coded types
2. **Performance Charts**: Real-time metrics and statistics
3. **Q-Table Display**: Live Q-table analysis for Nash Q agents
4. **Kill Notifications**: Real-time hunt success alerts
5. **Status Dashboard**: Current simulation state and agent counts

#### **Agent Visual Representation**:
- **RandomHunter**: Yellow circles
- **GreedyHunter**: Light yellow circles
- **NashQHunter**: Red circles (larger)
- **MinimaxHunter**: Dark red squares (larger)
- **Prey**: Blue circles
- **NashQPrey**: Green circles (larger)
- **MinimaxPrey**: Dark green squares (larger)

### Q-Table Analysis (`QTableDisplayer.py`)

Provides comprehensive Q-table analysis:
- **Terminal Display**: Formatted Q-table output in console
- **Web Interface**: Real-time Q-table visualization
- **Learning Progress**: Q-value evolution tracking
- **State-Action Analysis**: Detailed learning behavior insights

## Configuration and Parameters

### Simulation Parameters (Configurable via UI)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `N_hunters` | Integer | 0-10 | 0 | Number of random hunters |
| `N_greedy_hunters` | Integer | 0-10 | 0 | Number of greedy hunters |
| `N_nash_q_hunters` | Integer | 0-10 | 1 | Number of Nash Q-Learning hunters |
| `N_minimax_hunters` | Integer | 0-10 | 0 | Number of Minimax hunters |
| `N_preys` | Integer | 0-20 | 0 | Number of basic prey |
| `N_nash_q_preys` | Integer | 0-20 | 1 | Number of Nash Q-Learning prey |
| `N_minimax_preys` | Integer | 0-20 | 0 | Number of Minimax prey |
| `minimax_search_depth` | Integer | 1-6 | 4 | Depth of Minimax tree search |
| `width` | Integer | 1-20 | 4 | Grid width |
| `height` | Integer | 1-20 | 4 | Grid height |

### Learning Parameters

#### **Nash Q-Learning**:
- **Learning Rate (α)**: 0.1 (how quickly agents learn)
- **Discount Factor (γ)**: 0.9 (importance of future rewards)
- **Exploration Rate (ε)**: 0.3-0.9 (exploration vs exploitation balance)
- **Epsilon Decay**: 0.995-0.998 (exploration reduction over time)

#### **Reward Structure**:
- **Hunter Successful Kill**: +10.0
- **Hunter Movement Cost**: -0.1
- **Prey Survival**: +0.1
- **Prey Death Penalty**: -10.0

## Technical Implementation Details

### Dependencies and Requirements

- **Python 3.7+**
- **Mesa**: Multi-agent modeling framework
- **Solara**: Web-based visualization interface
- **NumPy**: Numerical computations
- **Logging**: Comprehensive debugging and monitoring

### Performance Considerations

1. **Minimax Search Depth**: Higher depth provides better strategy but increases computation time
2. **Grid Size**: Larger grids allow more complex behaviors but reduce interaction frequency
3. **Agent Population**: More agents increase interaction complexity but may slow simulation
4. **Nash Q-Learning Synchronization**: Ensures proper learning but adds computational overhead

### Logging and Debugging

The project includes comprehensive logging:
- **Agent Actions**: Detailed movement and decision logging
- **Q-Learning Updates**: Q-value changes and learning progress
- **Kill Events**: Hunt success tracking with positions and agents involved
- **Performance Metrics**: Real-time statistics and analysis

## Research and Educational Applications

### Multi-Agent Systems Research

This project serves as a testbed for:
- **Game-Theoretic Learning**: Nash Q-Learning in competitive environments
- **Algorithm Comparison**: Direct performance comparison between AI approaches
- **Emergent Behavior**: Complex interactions between different agent types
- **Learning Convergence**: Analysis of Q-Learning convergence in multi-agent settings

### Educational Value

- **Reinforcement Learning**: Practical implementation of Q-Learning variants
- **Game Theory**: Nash equilibrium computation and application
- **Search Algorithms**: Minimax with alpha-beta pruning
- **Multi-Agent Coordination**: Synchronization and interaction patterns
- **Visualization**: Real-time AI behavior observation and analysis

## Future Extensions and Research Directions

### Potential Enhancements

1. **Additional Algorithms**: 
   - Deep Q-Networks (DQN)
   - Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
   - Cooperative learning algorithms

2. **Environment Complexity**:
   - Obstacles and terrain features
   - Dynamic environments
   - Resource collection mechanics

3. **Advanced Metrics**:
   - Convergence analysis tools
   - Strategy evolution tracking
   - Cooperation/competition balance measurement

4. **Performance Optimization**:
   - Parallel agent execution
   - GPU acceleration for learning
   - Distributed simulation support

## Getting Started

### Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd AAMAS-Project
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**:
   ```bash
   pip install mesa solara numpy
   ```

4. **Run the Simulation**:
   ```bash
   cd src
   python run.py
   ```

5. **Access Web Interface**:
   Open your browser and navigate to the provided local URL (typically `http://localhost:8765`)

### Configuration and Experimentation

1. Use the web interface sliders to configure agent populations
2. Adjust grid size for different interaction patterns
3. Modify Minimax search depth for performance vs. strategy trade-offs
4. Monitor Q-tables for learning progress analysis
5. Compare performance metrics between different agent types

## Research Context and AAMAS Conference

This project appears to be developed for the **AAMAS (Autonomous Agents and Multiagent Systems)** conference, which is a premier venue for research in:

- Multi-agent systems
- Game theory applications
- Reinforcement learning
- Artificial intelligence
- Distributed problem solving

The implementation demonstrates practical applications of theoretical concepts in multi-agent learning and provides a platform for empirical research in competitive multi-agent environments.

---

**Authors**: AASMA Project Team  
**Framework**: Mesa Multi-Agent Modeling  
**License**: [Add appropriate license]  
**Last Updated**: [Current Date]
