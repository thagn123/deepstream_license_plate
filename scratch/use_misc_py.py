import re
with open('test_laplacian.py', 'r') as f:
    content = f.read()

python_meta_logic = """                # Retrieve Laplacian score from user_meta
                cv2_score = 0.0
                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                        if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("Laplacian.Score"):
                            meta_data_ptr = user_meta.user_meta_data
                            import ctypes
                            cv2_score = ctypes.cast(pyds.get_ptr(meta_data_ptr), ctypes.POINTER(ctypes.c_double)).contents.value
                            break
                    except StopIteration:
                        break
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break"""

content = content.replace(python_meta_logic, '                cv2_score = float(obj_meta.misc_obj_info[0])')

with open('test_laplacian.py', 'w') as f:
    f.write(content)
