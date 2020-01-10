import sys
import subprocess
import os

path = sys.argv[1]
path = path + "/"

output = sys.argv[2]

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    path + "render_%d_0.png", "-vb", "20M",
    output])