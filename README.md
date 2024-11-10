# Face finder 

A repository to sort the photos of a folder by faces, using dlib pretrained models. The data processing part of the code was greatly helped by [this repository](https://github.com/anisayari/easy_facial_recognition).

# How to use 

## Dependencies

Make sure that :

- dlib
- numpy
- pillow

are installed. Note that dlib requires a CMake program from cmake.org. To check which version you have, use `which cmake`. If the answer is not of shape `~/c/Program Files (x86)/bin/cmake`, you can `pip uninstall cmake` and install it again from cmake.org. Try restarting your computer if the `which cmake` command still give another answer.

## The code itself

- Put all the faces you want to detect in the "faces" folder, format must be jpg or png.
- Put all the images you want to analyze in the "to_process" folder.
- To change the upsampling and model, run : 
    - `path/to/file.py -u upsample_number -m model_name`
    - defaults are upsample = 1 and model is HOG+ linear SVM
- To change the folder of export from the default "processed" folder, add `-e path_to_folder`.
- After execution is finished, the pictures should be stored in subfolders named after the faces' names.
