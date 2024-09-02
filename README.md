# Auto-GeoGuessr: Enabling Knowledge Retrieval for Vision Tasks with Agents

## Introduction
Agentic systems are capable of competitive performance in GeoGuessr, showcasing their ability to identify regions in street view images with impressive precision. The results reveal that better reasoning from agents (notably a ReAct agent) improves performance but also increases costs and the likelihood of reaching recursion limits. Similarly, the use of more tools enhances performance but raises costs and the chances of reaching recursion limits.

This work demonstrates that incorporating agents with CV models allows for knowledge retrieval that can ground an image in the relevant context. As the next generation of MMLMs and LVMs are trained, it will be increasingly important for the training data to account for the invisible cultural, geographical, and historical contexts that provide a depth of meaning to an image beyond the immediately visible objects.

## Repository Structure
- `app/`: Contains the FastAPI app used to run the project
- `data/`: Contains the data used for the project (manually collected from GeoGuessr screenshots)
- `src/`:
    - `run_all_agents.sh`: Run inference for all agents on the test set
    - `inference.py`: Run inference for a given agent on a single location or test set
    - `eval.ipynb`: Evaluate the performance of all agents, view results in a series of plots
    - `agents/`: Contains the agents used for the project
    - `prompts/`: Contains the prompts used for each agent
    - `results/`: Contains the results from running each agent
    - `utils/`: Contains the helper functions used across the project
- `Dockerfile`: File used to containerize the project
- `docker-compose.yml`: File used to run the project

## How To Run Application
1. Clone the repo
2. Setup .env file with the following variables:
    - `LANGCHAIN_API_KEY`
    - `TAVILY_API_KEY`
    - `OPENAI_API_KEY`
    - `GPLACES_API_KEY`
    - `GOOGLE_API_KEY`
    - `GOOGLE_CSE_ID`
    - `LANGCHAIN_TRACING_V2`
3. Run `docker-compose build`
4. Run `docker-compose up`
4. Open browser and navigate to `localhost:8000`
5. Upload a streetview image, click "Upload Images", select which agent to use, click "Run Agent"
6. To try a different agent, clear the logs, switch the agent selection, and run again

## Discussion and Analysis
For full write-up, visit [Medium](https://medium.com/@j.calzaretta.ai/auto-geoguessr-enabling-knowledge-retrieval-for-vision-tasks-with-agents-9c5ba9cddb7f).
For evaluation metrics and plots, see src/results.ipynb
