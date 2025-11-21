# defines the tools and wires handlers

# # src/molml_mcp/tools/core_mol/__init__.py
# from .cleaning import clean_smiles_list
# from .smiles_ops import canonicalize_smiles, randomize_smiles

# def get_tools():
#     """
#     Return a list of MCP tool definitions.
#     Shape depends on your MCP server lib; here just a rough example.
#     """
#     return [
#         {
#             "name": "clean_smiles_list",
#             "description": "Clean a list of SMILES (remove invalids, standardize, etc.)",
#             "handler": clean_smiles_list,
#             "input_schema": {
#                 "type": "object",
#                 "properties": {
#                     "smiles": {"type": "array", "items": {"type": "string"}},
#                     "remove_salts": {"type": "boolean", "default": True},
#                 },
#                 "required": ["smiles"],
#             },
#         },
#         {
#             "name": "canonicalize_smiles",
#             "description": "Convert SMILES to canonical RDKit representation.",
#             "handler": canonicalize_smiles,
#             "input_schema": {
#                 "type": "object",
#                 "properties": {
#                     "smiles": {"type": "string"},
#                 },
#                 "required": ["smiles"],
#             },
#         },
#         {
#             "name": "randomize_smiles",
#             "description": "Generate randomized SMILES for data augmentation.",
#             "handler": randomize_smiles,
#             "input_schema": {
#                 "type": "object",
#                 "properties": {
#                     "smiles": {"type": "string"},
#                     "n": {"type": "integer", "default": 1, "minimum": 1},
#                 },
#                 "required": ["smiles"],
#             },
#         },
#     ]