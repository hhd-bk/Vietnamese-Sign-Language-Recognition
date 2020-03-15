import importlib
import os

modules = ['numpy', 'cv2', 'PIL', 'tensorflow', 'keras', 'imutils', 'scipy', 'argparse']
whl_files =["numpy-1.14.5-cp36-none-win_amd64.whl", 'opencv_python-3.4.1.15-cp36-cp36m-win_amd64.whl',
            'Pillow-5.1.0-cp36-cp36m-win_amd64.whl', 'tensorflow-1.8.0-cp36-cp36m-win_amd64.whl',
            'Keras-2.2.0-py2.py3-none-any.whl'     , 'imutils-0.4.6.tar.gz',
            'spicy-0.16.0-py2.py3-none-any.whl'    , 'argparse-1.4.0.tar.gz']

for i in modules:
    try:
        importlib.import_module(i)
        print('%s Installed' %i)
    except ImportError:
        print('%s is not installed' %i)
        print('Installing %s' %i)
        os.system('pip install modules\%s' %whl_files[whl_files.index(i)])

