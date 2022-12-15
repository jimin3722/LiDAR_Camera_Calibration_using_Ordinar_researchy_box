import pyrealsense2 as rs
import cv2 
import numpy as np 

def depth() :
    count = 0
    pipeline = rs.pipeline()
    config = rs.config()
    align_to = rs.stream.color
    align = rs.align(align_to)
    colorizer = rs.colorizer()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 5)

    while True:

        # Read frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frames = aligned_frames.get_depth_frame()

        # RGB camera
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frames.get_data())

        depth_color_image = np.asanyarray(colorizer.colorize(aligned_depth_frames).get_data())

        cv2.imshow('RGB', color_image)

        
        count+= 1
        if cv2.waitKey(5) == ord('q') :
            cv2.imwrite('test_img.png', color_image)
            break

if __name__ =='__main__' :
    depth()