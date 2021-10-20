import math
from USPP import main
import detspall
import os
import pandas as pd
import cv2
import shutil
import numpy as np

def mkNewDirs (dir='ref_frame'):
    if os.path.isdir(dir):
       try:
            shutil.rmtree(dir)
            os.mkdir(dir)
       except OSError as e:
           print("Error: %s : %s" % (dir, e.strerror))
    else:
        os.mkdir(dir)

def annotateImg (seg, ref_img, scale=0.5, roi=None, save=True):
    """
    Takes the output of semantic segmentation and tries to put it back into the original image.
    * cut off the sides of the segmentation by buffer number of pixels
    * upscale the segmentation... by 1/scale (which was used to downscale the image during segmentation)
    * place it in the correct location
    """
    buffer = 20
    row, col, _ = ref_img.shape
    full_mask = np.zeros((row, col))
    seg = seg[buffer:-buffer, buffer:-buffer]
    row_seg, col_seg = seg.shape
    mask = cv2.resize(seg.astype('float32'), (int(col_seg/scale), int(row_seg/scale)), interpolation=cv2.INTER_AREA)
    full_mask[int(roi[0]+buffer):int(roi[0]+buffer+row_seg/scale), int(roi[1]+buffer):int(roi[1]+buffer+col_seg/scale)] = mask
    if save:
        cv2.imwrite('full_mask.png', full_mask.astype('uint8')*255)
    return full_mask

if __name__ == "__main__":
    # num_images
    num_images = 8

    # set image data directory (replace with image stream)
    DATADIR = '../YOLOX/datasets/ig_sim_alt_texture/'
    lst = os.listdir(DATADIR)

    # cache dataframe
    bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'track'])

    # create instance of the spalling detector
    spallDetector = detspall.detspall(model='yolox.onnx')

    # loop through images.
    # For this demo segment the first spalling or where track == 1
    # when track > 1, break and do unsupervised segmentation
    lastTrackIdx = 1
    for i in range(len(lst)):
        imgFile = os.path.join(DATADIR, str(i) + '_rgb.jpg')
        img = cv2.imread(imgFile, -1)
        bbsTrack = spallDetector.detAndTrack(img)

        if bbsTrack.size > 0: #check if returned a bounding box
            bbsTrack = bbsTrack[0]
            #print(bbsTrack)
            if bbsTrack[4] == lastTrackIdx: #still tracking same object
                entry = pd.DataFrame(
                        {'file': imgFile, 'x1': bbsTrack[0], 'y1': bbsTrack[1], 'x2': bbsTrack[2], 'y2': bbsTrack[3], 'track': bbsTrack[4]}, index=[i])
                bb_df = bb_df.append(entry)
                # real implementation we will need to save the images in a different directory
            else: #starting to track the next object (TRIGGER)
                lastTrackIdx = bbsTrack[4]
                # need to sample the dataframe for number of images
                bb_df_elements = bb_df.sample(n=num_images) #randomly sample 8 elements
                break

    # Using the identified bb_df_elements apply USPP
    # create some directories (clean wipe)
    mkNewDirs('ref_frame')
    mkNewDirs('sub_frames')
    mkNewDirs('aug')

    # Save images from dataframe to folders
    # Let the middle image be the reference frame
    ref_num = math.floor(num_images/2)
    counter = 0
    # save the reference img
    ref_img = None
    # copy the row of the dataframe
    ref_row_df = None

    # Read in images in the samples dataframe
    # Save the images into their respective folders
    for index, row in bb_df_elements.iterrows():
        img_temp = cv2.imread(row['file'])
        if counter == ref_num:
            ref_img = img_temp
            cv2.imwrite(os.path.join('ref_frame', str(index)+'.png'), img_temp)
            ref_row_df = row
        else:
            cv2.imwrite(os.path.join('sub_frames', str(index)+'.png'), img_temp)
        counter = counter+1

    # Run pre-processing
    roi_width = ref_row_df['x2']-ref_row_df['x1']
    roi_height = ref_row_df['y2']-ref_row_df['y1']

    roi_x1 = ref_row_df['x1'] + ((roi_width-1.5*roi_width)/2)
    roi_y1 = ref_row_df['y1'] + ((roi_height-1.5*roi_height)/2)

    main.prepDefect(REF_DIR='ref_frame', SUB_DIR='sub_frames', AUG_DIR='aug', scale=0.5, roi=np.array([roi_x1, roi_y1, 1.5*roi_width, 1.5*roi_height]))

    # run unsupervised segmentation
    mask = main.scribbsDefect(AUG_DIR='aug', scribbs=None, save=True)
    full_mask = annotateImg(mask, ref_img, scale=0.5, roi=[roi_x1, roi_y1, roi_width, roi_height], save=True)