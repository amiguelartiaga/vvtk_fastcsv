"""Parse CSV straight into a torch tensor, optionally landing on the GPU."""
import torch

import vvtk_fastcsv as fc

x = torch.randn(500_000, 16)
fc.write_csv(x, "features.csv")  # row-major tensor memory is written directly

t = fc.read_torch("features.csv")                      # float32 tensor, parsed in place
t64 = fc.read_torch("features.csv", dtype=torch.float64)  # every cell parsed directly as float64
print(t.shape, t.dtype, t64.dtype, torch.allclose(t, x))

if torch.cuda.is_available():
    # Parsed into pinned host memory, then one asynchronous copy to the device.
    g = fc.read_torch("features.csv", device="cuda")
    print(g.device)

# Preallocated target (for instance a pinned buffer reused by a DataLoader):
buf = torch.empty((600_000, 16), pin_memory=torch.cuda.is_available())
view = fc.read_csv("features.csv", output="torch", out=buf)
print(view.shape, view.data_ptr() == buf.data_ptr())
