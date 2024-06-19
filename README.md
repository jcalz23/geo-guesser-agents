## Auto-GeoGuessr: Enabling Knowledge Retrieval for Vision Tasks with Agents
For full write-up, visit [Medium](https://medium.com/@j.calzaretta.ai/auto-geoguessr-enabling-knowledge-retrieval-for-vision-tasks-with-agents-9c5ba9cddb7f).

For evaluation metrics and plots, see src/results.ipynb

Agentic systems are capable of competitive performance in GeoGuessr, showcasing their ability to identify regions in street view images with impressive precision. The results reveal that better reasoning from agents (notably a ReAct agent) improves performance but also increases costs and the likelihood of reaching recursion limits. Similarly, the use of more tools enhances performance but raises costs and the chances of reaching recursion limits.

This work demonstrates that incorporating agents with CV models allows for knowledge retrieval that can ground an image in the relevant context. As the next generation of MMLMs and LVMs are trained, it will be increasingly important for the training data to account for the invisible cultural, geographical, and historical contexts that provide a depth of meaning to an image beyond the immediately visible objects.