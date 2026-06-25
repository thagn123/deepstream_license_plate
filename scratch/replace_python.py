import re

with open('test_laplacian.py', 'r') as f:
    content = f.read()

# Remove align_plate function
content = re.sub(r'def align_plate\(.*?\n        return plate_crop, False\n', '', content, flags=re.DOTALL)

# Add laplacian element to pipeline
if 'dslaplacian' not in content:
    # Find pipeline.add and insert dslaplacian
    content = content.replace('sink = Gst.ElementFactory.make("fakesink", "fake-output")', 'sink = Gst.ElementFactory.make("fakesink", "fake-output")\n    laplacian = Gst.ElementFactory.make("dslaplacian", "laplacian")\n    laplacian.set_property("class-id", 13)')
    content = content.replace('for el in [source, nvvidconv, streammux, pgie, tiler, nvvidconv_rgba, caps_gpu, sink]:', 'for el in [source, nvvidconv, streammux, pgie, laplacian, tiler, nvvidconv_rgba, caps_gpu, sink]:')
    
    # Link laplacian
    content = content.replace('pgie.link(tiler)', 'pgie.link(laplacian)\n    laplacian.link(tiler)')

# Modify the probe
probe_logic = """def sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == 13:
                w = int(obj_meta.rect_params.width)
                h = int(obj_meta.rect_params.height)
                
                # Retrieve Laplacian score from user_meta
                cv2_score = 0.0
                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                        if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("Laplacian.Score"):
                            cv2_score = pyds.get_ptr_double(user_meta.user_meta_data)
                            break
                    except StopIteration:
                        break
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break
                
                if w < 100 or h < 30:
                    category = "1_qua_nho_bo_qua"
                elif cv2_score < 300:
                    category = "2_du_to_nhung_mo_hoac_vo_hat"
                elif cv2_score < 700:
                    category = "3_du_to_vua_net"
                else:
                    category = "4_du_to_rat_net"

                if save_counters.get(category, 0) < 30:
                    print(f"[{category}] Biển số {w}x{h}: Điểm Laplacian từ GPU = {cv2_score:.1f}")
                    save_counters[category] = save_counters.get(category, 0) + 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK"""

content = re.sub(r'def sink_pad_buffer_probe\(pad, info, u_data\):.*return Gst\.PadProbeReturn\.OK', probe_logic, content, flags=re.DOTALL)

with open('test_laplacian.py', 'w') as f:
    f.write(content)
