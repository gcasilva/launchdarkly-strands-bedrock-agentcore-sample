Sample agent using LaunchDarkly AI Configs SDK + Strands SDK deployed on Amazon Bedrock Agentcore.

Pre-requisites in your local computer:
1 - Python,2 - AWS CLi (latest version),3 - strands-agents - The Strands Agents SDK (https://strandsagents.com/latest/)

To run:

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

Install the following required packages:

bedrock-agentcore - The Amazon Bedrock AgentCore SDK for building AI agents
bedrock-agentcore-starter-toolkit - The Amazon Bedrock AgentCore starter toolkit

pip install bedrock-agentcore strands-agents bedrock-agentcore-starter-toolkit
pip install -r requirements.txt

Verify installation of Bedrock Agentcore SDK:
agentcore --help

Configure your Bedrock Agentcore Agent:
agentcore configure -e my_agent.py -r <YOUR-AWS-REGION>

Deploy your Bedrock Agentcore Agent:
agentcore launch

Test your deployed agent:
agentcore invoke '{"prompt": "Hello"}'
