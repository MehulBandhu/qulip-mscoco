import torch, glob
init = sorted(glob.glob("checkpoints/mscoco-vqc10k-drop/*/*/init.pt"))
best = sorted(glob.glob("checkpoints/mscoco-vqc10k-drop/*/*/best.pt"))
if not init:
    print("no init.pt that run predates the snapshot patch"); raise SystemExit
a = torch.load(init[-1], map_location="cpu")["text"]["params"]
b = torch.load(best[-1], map_location="cpu")["text"]["params"]
n_sym = 338953
print(f"table  slots moved: {(a[:n_sym] - b[:n_sym]).abs().gt(1e-8).float().mean():.1%}")
print(f"fallback moved:     {(a[n_sym+1:] - b[n_sym+1:]).abs().gt(1e-8).float().mean():.1%}")
print(f"fallback max delta: {(a[n_sym+1:] - b[n_sym+1:]).abs().max():.2e}")
