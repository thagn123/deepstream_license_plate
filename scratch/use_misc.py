import re
with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'r') as f:
    content = f.read()

# Replace user_meta logic
user_meta_logic = """          // Attach result
          NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
          double *meta_data = (double *)g_malloc0(sizeof(double));
          *meta_data = variance;
          
          user_meta->user_meta_data = (void *)meta_data;
          user_meta->base_meta.meta_type = nvds_get_user_meta_type((gchar*)"Laplacian.Score");
          user_meta->base_meta.release_func = [](gpointer data, gpointer user_data) {
              NvDsUserMeta *meta = (NvDsUserMeta *)data;
              if (meta->user_meta_data) {
                  g_free(meta->user_meta_data);
                  meta->user_meta_data = nullptr;
              }
          };
          
          nvds_add_user_meta_to_obj(obj_meta, user_meta);"""

content = content.replace(user_meta_logic, 'obj_meta->misc_obj_info[0] = (int)variance;')

with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'w') as f:
    f.write(content)
