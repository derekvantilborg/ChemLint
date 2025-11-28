# from .tools import get_all_tools

# src/molml_mcp/server.py


# server.py
from mcp.server.fastmcp import FastMCP
from molml_mcp.tools import canonicalize_smiles

# All tools we want to expose via the MCP server
from molml_mcp.resources.logistics import get_all_resource_functions



# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add resource management tools
for tool_func in get_all_resource_functions():
    mcp.add_tool(tool_func)


# Add tools
mcp.add_tool(canonicalize_smiles)


