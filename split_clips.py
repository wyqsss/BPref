import subprocess
import os
import ffmpeg
from time import strftime
from time import gmtime
import ffmpy

def split_video(video_path, output_path):
        # video_path = f"/home/wyq/language-rl/videos/evaluation/E_eval_{j}_{k}.mp4"

        # output_path = "/home/wyq/language-rl/videos/evaluation/clips"
        # cmdline = "ffmpeg -i input_audio.mp4 -ss 00:00:30 -t 10 output_audio.mp4 "

        # cmdline = ('ffmpeg', '-i', video_path, '-ss', begin_time, '-t', 3, out_video)

        # if hasattr(os,'setsid'): #setsid not present on Windows
        #     proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        # else:
        #     proc = subprocess.Popen(cmdline, stdin=subprocess.PIPE)
    print(video_path)
    output_path = os.path.join(output_path, video_path.split('/')[-1][:-4])
    os.makedirs(output_path, exist_ok=True)
    result = ffmpeg.probe(video_path)['format']['duration']
    print(result)
    for i in range(int(float(result)) // 1):
        start_time = strftime("%H:%M:%S", gmtime(i*1))
        print(f"start_time is {start_time}")
        out_video = os.path.join(output_path, f"{video_path.split('/')[-1][:-4]}_c{i}.mp4")
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
for i in range(10):
    split_video(f"videos/MiniGrid-DoorKey-5x5-v0_{i}.mp4", "videos/clips")

