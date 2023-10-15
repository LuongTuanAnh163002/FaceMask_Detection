from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
from pathlib import Path
import argparse
from utils import increment_path
import time
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect_and_predict_mask(frame, gray_img, model_predict):
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) > 0:
        (x, y, w, h) =  faces[0][0], faces[0][1], faces[0][2], faces[0][3]
    else:
        return (0, 0), (0, 0, 0, 0)
    

    frame = frame[y:y+h, x:x+h]
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    faces = np.array(face, dtype="float32")
    faces = np.expand_dims(faces, axis = 0)
    preds = model_predict.predict(faces, batch_size=32)
    return preds, (x, y, w, h)

def detect(opt):
    save_dir, weights, demo, source = opt.save_dir, opt.weights, opt.demo, opt.source
    model = load_model(weights)
    if demo == 0:
        video = cv2.VideoCapture(0)
        while True:
            ret, img = video.read()
            print("Read image")
            print(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            prediction, location = detect_and_predict_mask(img, gray, model)
            if location != (0, 0, 0, 0):
                (non_mask, mask) = prediction[0][0], prediction[0][1]
                if mask > non_mask:
                    (x, y, w, h) = location
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = "Mask"
                    cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                else:
                    (x, y, w, h) = location
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = "Non-Mask"
                    cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            else:
                (x, y, w, h) = location
                cv2.putText(img, "No face", (x+100, y+100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            
            cv2.imshow("Face detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        video.release()
        cv2.destroyAllWindows()
    elif demo == 1:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        video = cv2.VideoCapture(source)
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = save_dir + "/" + "output.mp4"
        vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        t0 = time.time()
        while True:
            ret, img = video.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                prediction, location = detect_and_predict_mask(img, gray, model)
                if location != (0, 0, 0, 0):
                    (non_mask, mask) = prediction[0][0], prediction[0][1]
                    if mask > non_mask:
                        (x, y, w, h) = location
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text = "Mask"
                        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    else:
                        (x, y, w, h) = location
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text = "Non-Mask"
                        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                else:
                    (x, y, w, h) = location
                    cv2.putText(img, "No face", (x+100, y+100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

                vid_writer.write(img)
            else:
                break
        
        print(f"Results saved to {save_dir}")
        print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.h5', help='model.h5 path(s)')
    parser.add_argument('--source', type=str, default='inference/video', help='source')  # file path of video
    parser.add_argument('--demo', type=int, default=0, help='video or webcam')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    detect(opt)