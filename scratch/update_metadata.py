import re
with open('src/lpr/probes/metadata.py', 'r') as f:
    content = f.read()

# Add laplacian filter check
laplacian_check = """            if p.rect_params.width < lpr_state.min_plate_width or p.rect_params.height < lpr_state.min_plate_height:
                continue

            # Laplacian Filter
            if not getattr(lpr_state, 'disable_laplacian', False):
                lap_score = int(p.misc_obj_info[0])
                if lap_score < 300 and lap_score > 0:
                    continue  # Skip blurry plate"""
content = content.replace('            if p.rect_params.width < lpr_state.min_plate_width or p.rect_params.height < lpr_state.min_plate_height:\n                continue', laplacian_check)

with open('src/lpr/probes/metadata.py', 'w') as f:
    f.write(content)
