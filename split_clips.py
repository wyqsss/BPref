import subprocess
import os
import ffmpeg
from time import strftime
from time import gmtime
import ffmpy

def split_video(video_path, output_path, nc):
        # video_path = f"/home/wyq/language-rl/videos/evaluation/E_eval_{j}_{k}.mp4"

        # output_path = "/home/wyq/language-rl/videos/evaluation/clips"
        # cmdline = "ffmpeg -i input_audio.mp4 -ss 00:00:30 -t 10 output_audio.mp4 "

        # cmdline = ('ffmpeg', '-i', video_path, '-ss', begin_time, '-t', 3, out_video)

        # if hasattr(os,'setsid'): #setsid not present on Windows
        #     proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        # else:
        #     proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE)
    print(video_path)
    # output_path = video_path.split("/")[:-1]
    # output_path = os.path.join(output_path,  f"clips_e{nc}")
    # os.makedirs(output_path, exist_ok=True)
    result = ffmpeg.probe(video_path)['format']['duration']
    print(result)
    for i in range(int(float(result)) // 1):
        start_time = strftime("%H:%M:%S", gmtime(i*1))
        print(f"start_time is {start_time}")
        # out_video = os.path.join(output_path, f"{video_path.split('/')[-1][:-4]}_c{i}.mp4")
        out_video = os.path.join(output_path, f"v_{nc}_clip_{i}.mp4")
        ff = ffmpy.FFmpeg(
            inputs={video_path: None},
            outputs={out_video: [
                '-ss', start_time,
                '-t', '1',
                # '-vcodec', 'copy',
                # '-acodec', 'copy'
            ]}
        )
        # print(ff.cmd)
        ff.run()
        # break
# for i in range(10):
#     split_video(f"videos/MiniGrid-DoorKey-5x5-v0_{i}.mp4", "videos/clips")
# split_video("videos/metaworld_door-open-v2checkpoints/metaworld_door-open-v2/seed-6789-step-52000/rl-video-episode-0.mp4", "videos/metaworld_door-open-v2checkpoints/metaworld_door-open-v2/seed-6789-step-52000")

bas_path = "p2video/metaworld_door-open-v2checkpoints/metaworld_door-open-v2"
dirs = os.listdir(bas_path)
count = 0
for s in dirs:
    spath = os.path.join(bas_path, s)
    for v in os.listdir(spath):
        if v.split(".")[-1] == "mp4":
            video_path = os.path.join(spath, v)
            split_video(video_path=video_path, output_path="p2dataset", nc=count)
            count += 1