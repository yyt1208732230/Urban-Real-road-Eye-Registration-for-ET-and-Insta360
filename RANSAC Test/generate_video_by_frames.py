import os
import cv2

def sort_frame_numbers(filename):
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except ValueError:
        return 0

def merge_images_to_video(image_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith("frame_")]
    images.sort(key=sort_frame_numbers)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

# Example usage
newvideo_name = "ET_insta_registra_test_P14.mp4"
image_folder = "./output/ET_Insta_Registra/"
output_video_path = "./output/reigstra_video/" + newvideo_name
merge_images_to_video(image_folder, output_video_path, fps=30)
