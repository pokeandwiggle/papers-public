"""Autoregressive action-token policies (OAT and QueST tokenizers).

Ported from the OAT reference implementation (https://github.com/Chaoqi-LIU/oat,
revision 1da9269). OAT and QueST share the observation encoder and the
autoregressive backbone; only the action tokenizer differs.
"""
