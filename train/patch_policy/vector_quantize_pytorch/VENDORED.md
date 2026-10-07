# Vendored residual vector quantizer

Copied from the Patch Policy reference implementation
(`models/vq_behavior_transformer/vector_quantize_pytorch`,
https://github.com/gaoyuezhou/patch_policy at commit
`55ccb4f705d465560e332ff6bd14d9d6c9b26cea`, MIT license), itself derived from
lucidrains' `vector-quantize-pytorch`. The only change replaces the deprecated
`torch.cuda.amp.autocast` import with the equivalent `torch.amp.autocast("cuda")`.
