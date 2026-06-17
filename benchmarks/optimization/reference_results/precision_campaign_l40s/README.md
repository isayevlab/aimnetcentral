# Precision Campaign L40S Reference Artifacts

These files were borrowed from the historical `test/aimnet-precision-campaign`
branch only as comparison data for GPU-throughput work. They are not generated
by this optimization branch and should not be interpreted as evidence for any
production precision API.

The artifact metadata records commit `8092749`, while the precision campaign
branch head used as the source for these files is `e030030`. Treat that
provenance mismatch as part of the historical record. New benchmark runs from
`opt/gpu-throughput` should supersede these references and include their own
full SHA, branch, dirty flag, command, device, Torch/CUDA, nvalchemiops, and
warp metadata.
