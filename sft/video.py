import sys
import subprocess
import os

path = sys.argv[1]
path = path + "/"

output = sys.argv[2]

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    path + "texture_%d.png", "-vb", "20M",
    output])