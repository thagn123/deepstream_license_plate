import re
with open('src/lpr/pipeline.py', 'r') as f:
    content = f.read()

# state.disable_laplacian = cfg["disable_laplacian"]
content = content.replace('state.kafka_enabled = cfg["kafka_enable"]', 'state.disable_laplacian = cfg["disable_laplacian"]\n    state.kafka_enabled = cfg["kafka_enable"]')

# Conditionally add/link laplacian
add_replacement = """    if not state.disable_laplacian:
        for el in [pgie, tracker, laplacian, sgie3, tiler, nvvidconv, caps_gpu, nvosd, tee, queue_display, sink]:
            pipeline.add(el)
    else:
        for el in [pgie, tracker, sgie3, tiler, nvvidconv, caps_gpu, nvosd, tee, queue_display, sink]:
            pipeline.add(el)"""
content = re.sub(r'    for el in \[pgie, tracker, laplacian, sgie3, tiler, nvvidconv, caps_gpu, nvosd, tee, queue_display, sink\]:\n        pipeline.add\(el\)', add_replacement, content)

link_replacement = """    streammux.link(pgie)
    pgie.link(tracker)
    if not state.disable_laplacian:
        tracker.link(laplacian)
        laplacian.link(sgie3)
    else:
        tracker.link(sgie3)"""
content = re.sub(r'    streammux.link\(pgie\)\n    pgie.link\(tracker\)\n    tracker.link\(laplacian\)\n    laplacian.link\(sgie3\)', link_replacement, content)

with open('src/lpr/pipeline.py', 'w') as f:
    f.write(content)
