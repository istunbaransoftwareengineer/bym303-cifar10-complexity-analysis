import sys, time, torch, torch.nn as nn
sys.path.insert(0, sys.argv[1])
from models import build_model
torch.backends.cudnn.benchmark = True
dev = "cuda"
for name in ["ResNet18", "MobileNetV2", "MediumCNN"]:
    for cl in [True, False]:
        for dtype in [torch.float16, torch.bfloat16]:
            m = build_model(name).to(dev)
            if cl:
                m = m.to(memory_format=torch.channels_last)
            opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
            scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
            x = torch.randn(128, 3, 32, 32, device=dev)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            y = torch.randint(0, 10, (128,), device=dev)
            lf = nn.CrossEntropyLoss()
            def step():
                with torch.autocast("cuda", dtype=dtype):
                    loss = lf(m(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            for _ in range(15): step()
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(50): step()
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50
            print(f"{name:12s} cl={cl!s:5s} {str(dtype):15s} {dt*1000:7.2f} ms/iter -> {dt*391:6.1f} s/epoch", flush=True)
