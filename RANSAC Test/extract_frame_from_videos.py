import os
import cv2
import ffmpeg

def extract_frames_from_video(video_path, frame_folder, target_fps=30):
    os.makedirs(frame_folder, exist_ok=True)

    # Get the first frame from the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Get the video's frame rate
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Calculate the number of frames to duplicate or remove
    frame_rate_ratio = target_fps / video_fps

    # Interpolate frames using ffmpeg if needed
    if frame_rate_ratio != 1.0:
        output_video_path = os.path.join(frame_folder, "interpolated.mp4")
        ffmpeg.input(video_path).filter('minterpolate', fps=target_fps).output(output_video_path).run(overwrite_output=True)
        video_path = output_video_path

    # Extract frames at the target FPS
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame
        frame_filename = os.path.join(frame_folder, f"frame_{frame_num:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_num += 1

    cap.release()

# Specify the paths to the input videos and the frame folders
video1_path = "F:/LaurenceHub/video_image_registration/P14/P14-L3-test.mp4"
video2_path = "F:/LaurenceHub/video_image_registration/P14/P14-L3-test-insta.mp4"
video3_path = "F:/LaurenceHub/video_image_registration/P14/P14-L3-baseline-test.mp4"
frame_folder1 = "./output/ET_video_frame"
frame_folder2 = "./output/Insta_video_frame"
frame_folder3 = "./output/bl_video_frame"

# Extract frames from video 1
extract_frames_from_video(video1_path, frame_folder1, target_fps=30)

# Extract frames from video 2
extract_frames_from_video(video2_path, frame_folder2, target_fps=30)
extract_frames_from_video(video3_path, frame_folder3, target_fps=30)
