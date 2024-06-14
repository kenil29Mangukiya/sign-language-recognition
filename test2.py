import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk


class HandGestureRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Hand Gesture Recognition")

        # Initialize OpenCV variables
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=2)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ["1", "2", "3", "4", "5", "2", "7", "8", "9", "A", "4,B", "C", "1,D", "E", "3,F", "G", "H",
                     "2,I", "I FEEL YOU", "I LOVE YOU", "J", "2,VICTORY", "L", "M", "N", "O", "P", "Q", "R", "S",
                       "T", "THANK YOU", "U", "V", "4,W", "X", "U", "V", "W", "X", "Y", "Z"]

        # Create GUI elements
        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.pack()
        self.start_button = tk.Button(master, text="Start", fg="white", bg='green',
                                      font='Helvetica 12 bold italic', command=self.start_capture)
        self.start_button.pack()

    def start_capture(self):
        # Function to start capturing and processing frames
        def process_frame():
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1)  # Flip horizontally for better visualization
                imgOutput = img.copy()
                hands, img = self.detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                    imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = self.imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                        wGap = math.ceil((self.imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = self.imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                        hGap = math.ceil((self.imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                    cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                                  (x - self.offset + 90, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, self.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - self.offset, y - self.offset),
                                  (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)
                    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                    imgOutput = Image.fromarray(imgOutput)
                    imgOutput = ImageTk.PhotoImage(image=imgOutput)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgOutput)
                    self.canvas.imgOutput = imgOutput  # Prevent garbage collection
            self.master.after(10, process_frame)  # Call the function again after 10 milliseconds

        # Start capturing frames
        process_frame()


def main():
    root = tk.Tk()
    app = HandGestureRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# start_button = tk.Button(root, text="start", fg="white", bg="green", font="Helvetica 12 bold italic", command=cv2.imshow() , height="4", width="16", activebackground='lightblue' )
# start_button.grid(row= 3, column=2, pady=10, padx=20)
# update_frame()
# root.mainloop()
