# Hunter-Prey Nash Q-Learning Simulation - Installation Guide

Project for the Autonomous Agents and Multi-Agent Systems course by group 10.

## Quick Setup Instructions

### Prerequisites
- **Python 3.7+** (tested with Python 3.9+)
- **Windows OS** (required)
- **pip** package manager

### 1. Download Project
```bash
# Assuming "AAMAS-Project" as the folder name
cd AAMAS-Project
```

### 2. Create Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
# Install required packages
pip install mesa solara altair networkx matplotlib scipy numpy
```

### 4. Run the Simulation
```powershell
# Navigate to source directory
cd src

# Start the simulation
solara run run.py
```

### 5. Access the Interface
1. After running the command, you'll see output like:
   ```
   Solara server starting...
   * Running on http://localhost:8765
   ```
2. Open your web browser and go to: **http://localhost:8765**

## Basic Usage

- Use the **sliders** on the left panel to configure agent numbers and grid size
- Click **"Reset"** to apply new settings
- Click **"Play"** to start the simulation
- Monitor the **grid visualization**, **charts**, and **Q-table displays** in real-time

