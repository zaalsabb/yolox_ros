import cv2
import pandas as pd

import wrapper as detector
from sort import *


class det_spall():

    def __init__(self, model='yolox.onnx'):
        self.session = detector.open_sess(model=model)
        self.mot_tracker = Sort()
        self.bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'conf', 'track'])

    def detect (self, img):
        final_boxes, final_scores, final_cls_inds = detector.run(sess=self.session, img=img, visual=False)
        if final_boxes is not None:
            final_boxes = final_boxes[final_scores.argmax(), :]
            x1 = final_boxes[0]
            y1 = final_boxes[1]
            x2 = final_boxes[2]
            y2 = final_boxes[3]
            conf = final_scores[final_scores.argmax()]
        else:
            return None
        return [x1, y1, x2, y2, conf]

    def detAndTrack (self, img):
        # Get detections [x1, y1, x2, y2, conf]
        detections = self.detect(img)
        if detections is not None:
            track_bbs_ids = mot_tracker.update(np.array([detections]))
        else:
            track_bbs_ids = mot_tracker.update(np.empty((0, 5)))
        # track_bbs_ids [x1, y1, x2, y2, track_idx]
        return track_bbs_ids

if __name__ == "__main__":
    DATADIR = '../YOLOX/datasets/ig_sim_closeup/'
    lst = os.listdir(DATADIR)

    session = detector.open_sess(model='yolox.onnx')

    mot_tracker = Sort()

    bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'conf', 'track'])

    start = time.time()
    for i in range(len(lst)):
        imgFile = os.path.join(DATADIR, str(i) + '_rgb.jpg')
        img = cv2.imread(imgFile, -1)
        final_boxes, final_scores, final_cls_inds = detector.run(sess=session, img=img, visual=False)
        if final_boxes is not None:
            final_boxes = final_boxes[final_scores.argmax(), :]
            x1 = final_boxes[0]
            y1 = final_boxes[1]
            x2 = final_boxes[2]
            y2 = final_boxes[3]
            conf = final_scores[final_scores.argmax()]
            track_bbs_ids = mot_tracker.update(np.array([[x1, y1, x2, y2, conf]]))
            if track_bbs_ids.size > 0:
                entry = pd.DataFrame({'file': imgFile, 'x1': [x1], 'y1': [y1], 'x2': [x2], 'y2': [y2], 'conf': [conf],
                                      'track': track_bbs_ids[0, 4]}, index=[i])
            else:
                entry = pd.DataFrame(
                    {'file': imgFile, 'x1': [x1], 'y1': [y1], 'x2': [x2], 'y2': [y2], 'conf': [conf], 'track': False},
                    index=[i])

            bb_df = bb_df.append(entry)
        else:
            track_bbs_ids = mot_tracker.update(np.empty((0, 5)))
    end = time.time()
    ex_time = end - start

    print(f"Execution Time: {ex_time} s")

    bb_df.to_csv('dets.csv', index=False)
