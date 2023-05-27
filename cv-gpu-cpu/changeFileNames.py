import os
import tkinter as tk
from tkinter.filedialog import askdirectory


def changeFileNames(src_folder, dst_folder):
    """It will change all the files' name inside the folder to consecutive integers starting from 0 and save into the
    destination folder"""
    for count, filename in enumerate(os.listdir(src_folder)):
        dst = f"{str(count)}.png"
        src = f"{src_folder}/{filename}"
        dst = f"{dst_folder}/{dst}"
        os.rename(src, dst)


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    src_folder_path = askdirectory(title='Select Folder')  # shows dialog box and return the path
    print(src_folder_path)
    dst_folder_path = askdirectory(title='Save Folder')

    changeFileNames(src_folder_path, dst_folder_path)
