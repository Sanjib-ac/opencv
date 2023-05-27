import os
import tkinter as tk
from tkinter.filedialog import askdirectory
import cv2

# root = tk.Tk()
# root.withdraw()
# src_folder_path = askdirectory(title='Select Folder')  # shows dialog box and return the path
# print(src_folder_path)
# dst_folder_path = askdirectory(title='Save Folder')
#
#
# def changePixelSize(src_folder, dst_folder):
#     """Resized image files"""
#     dim = (256, 256)
#     for count, filename in enumerate(os.listdir(src_folder)):
#         resized = cv2.resize(cv2.imread(f"{src_folder}/{filename}"), dim, interpolation=cv2.INTER_AREA)
#         cv2.imwrite(f"{dst_folder}/{filename}", resized)
#
#         # dst = f"{str(count)}.png"
#         # src = f"{src_folder}/{filename}"
#         # dst = f"{dst_folder}/{dst}"
#         # os.rename(src, dst)


# changePixelSize(src_folder_path, dst_folder_path)
file = r'C:\Users\Sanjib\Pictures\VisionDemoImages\junk\i\0.png'

image = cv2.imread(file)
print(image.shape)
