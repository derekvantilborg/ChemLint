# The single connection hub between MCP and all domains

# # src/molml_mcp/tools/__init__.py
# from .core_mol import get_tools as get_core_mol_tools
# from .ml import get_tools as get_ml_tools
# from .chembl_api import get_tools as get_chembl_tools
# from .protein import get_tools as get_protein_tools

# def get_all_tools():
#     tools = []
#     tools.extend(get_core_mol_tools())
#     tools.extend(get_ml_tools())
#     tools.extend(get_chembl_tools())
#     tools.extend(get_protein_tools())
#     # later: tools.extend(get_new_subpackage_tools())
#     return tools