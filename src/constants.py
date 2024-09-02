# Search parameters
TOP_N_SEARCHES = 3

# Evaluation parameters
MAX_SCORE = 5000
MAX_D = 1000
ALL_RUNS_DIR = "../data/results/runs/all/"
TOP_RUNS_DIR = "../data/results/runs/top/"

## Agent parameters
AGENT_LIST = ["prescribed_chain", "single_agent", "react_agent", "multi_agent_supervisor"]

# Prescribed Chain Agent
PRESCRIBED_CHAIN_MODEL = "gpt-4o"
PRESCRIBED_CHAIN_TEMPERATURE = 0.2

# Single Agent
SINGLE_AGENT_MODEL = "gpt-4o"
SINGLE_AGENT_RECURSION_LIMIT = 12
SINGLE_AGENT_TEMPERATURE = 0.2

# Multi-Agent Supervisor
MULTI_AGENT_MEMBERS = ["Researcher", "Map_Searcher", "Street_Viewer"]
MULTI_AGENT_MODEL = "gpt-4o"
MULTI_AGENT_RECURSION_LIMIT = 10
MULTI_AGENT_TEMPERATURE = 0.2

# React Agent
REACT_MODEL = "gpt-4o"
REACT_TEMPERATURE = 0.2
REACT_RECURSION_LIMIT = 12

