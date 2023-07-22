# notes
'''
This file is used for handling anything image related.
I suggest handling the local file encoding/decoding here as well as fetching any external images.
'''

# package imports
import base64
import os

# # logo information
# cwd = os.getcwd()
# logo_path = os.path.join(cwd, 'assets', 'logos', 'logo_main.png')
# logo_tunel = base64.b64encode(open(logo_path, 'rb').read())
# logo_encoded = 'data:image/png;base64,{}'.format(logo_tunel.decode())

cwd = os.getcwd()
red_png = os.path.join(cwd, 'assets', 'image', 'red.JPG')
red_base64 = base64.b64encode(open(red_png, 'rb').read()).decode('ascii')
red_base64_encoded = 'data:image/png;base64,{}'.format(red_base64)

blue_png = os.path.join(cwd, 'assets', 'image', 'blue.JPG')
blue_base64 = base64.b64encode(open(blue_png, 'rb').read()).decode('ascii')
blue_base64_encoded = 'data:image/png;base64,{}'.format(blue_base64)
