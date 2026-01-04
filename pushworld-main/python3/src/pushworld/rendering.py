
import numpy as np
import cv2
def savergb(rgb_array, name):
    if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
        rgb_array = (rgb_array * 255).astype(np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, bgr_array)

def create_rgb_video_opencv(data, output_file='rgb_video.avi', fps=10):
    """
    Создает видео из RGB данных используя OpenCV
    """
    first_frame = data[0]
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for i, rgb_frame in enumerate(data):
        bgr_frame = cv2.cvtColor((rgb_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(bgr_frame, f'Frame: {i}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(bgr_frame)
    
    out.release()
    print(f"Video saved as {output_file}")