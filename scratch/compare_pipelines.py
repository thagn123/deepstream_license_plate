import json

def load_plates(path):
    plates = set()
    try:
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("event") == "final_plate_track":
                    text = data.get("plate", "")
                    if text:
                        plates.add(text)
    except: pass
    return plates

plates_new = load_plates("/home/thagn/projects/deepstream/outputs/debug_new.jsonl")
plates_old = load_plates("/home/thagn/projects/deepstream/outputs/debug_old.jsonl")

print(f"=== Đánh giá độ chính xác (Biển Số Bắt Được) ===")
print(f"Tổng số biển nhận diện được (Pipeline 1 - Có Laplacian): {len(plates_new)}")
print(f"Tổng số biển nhận diện được (Pipeline 2 - Không Lap): {len(plates_old)}\n")

all_plates = sorted(plates_new.union(plates_old))
print(f"{'Biển số':<15} | {'Pipeline 1 (Có Lap)':<25} | {'Pipeline 2 (Không Lap)':<25}")
print("-" * 65)
for p in all_plates:
    n = "CÓ" if p in plates_new else "[MISS]"
    o = "CÓ" if p in plates_old else "[MISS]"
    print(f"{p:<15} | {n:<25} | {o:<25}")

