import cv2

# videoname = "/home/wyq/deep-rl-action-advising/Runs/Videos/ALE50V0_000_240_20221029-134519/24/15_18753.mp4"

def cap_pic(video_name):
    capture = cv2.VideoCapture(video_name)
    i = 0
    if capture.isOpened():
        while True:
            ret,img=capture.read() # img 就是一帧图片            
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            print("read video")
            cv2.imwrite(f"captures/{i}.jpg", img)
            i += 1
            if not ret:break # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')

cap_pic("dataset/clip_0_0.mp4")