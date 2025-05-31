# Before running the script, configure LLM API keys and endpoints in .py files under AgentAuditor/tasks, along with model names
# To run an experiement, take rjudge as an example:

# Sequentially run the following commands, remember to check successful completion of each step
python -m AgentAuditor rjudge preprocess
python -m AgentAuditor rjudge cluster
python -m AgentAuditor rjudge demo
python -m AgentAuditor rjudge infer_emb
python -m AgentAuditor rjudge infer
python -m AgentAuditor rjudge eval

# Notes: Only one model and one dataset can be used at a time. If you want to parallelize the process,
# just make a copy of the repo.
