from subprocess import call

call(["ffmpeg", "-framerate", "24", "-i",
    "results/shadow_art/multitarget/final/iter_%d_t1.png", "-vb", "20M",
    "results/shadow_art/multitarget/final_1.mp4"])

call(["ffmpeg", "-framerate", "24", "-i",
    "results/shadow_art/multitarget/final/iter_%d_t0.png", "-vb", "20M",
    "results/shadow_art/multitarget/final_0.mp4"])
