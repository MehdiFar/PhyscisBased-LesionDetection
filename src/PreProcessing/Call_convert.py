import os
import Convert2Nifti
import files_directories as f_dirs


if __name__ == '__main__':
    args = [f_dirs.png_files_head_add +f'{i:02d}'+'/Images_png' for i in range(1,57)]
    # args = ['/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Images_png_'+f'{i:02d}'+'/Images_png' for i in range(1,57)]
    for arg in args:
        Convert2Nifti.conv2Nifti(arg)
