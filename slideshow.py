import os
import numpy as np
import cv2
from random import random

SHOW_DURATION_MS = 1000 * 10

IMAGE_PATH = "STABLE_DIFFUSION_IMAGE_PATH"

class StableDiffusionImage:
    def __init__(self, path:str, score:float = 100):
        self.path = path
        self.score = score
        self.parse_prompt()
    def load(self) -> np.ndarray:
        self.score = 0
        img = cv2.imread(self.path)
        overlay = img.copy()
        pos = (30, 30)
        alpha = .3
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .6
        font_thickness = 1
        text_color_fg = (255,255,255)
        text_color_bg = (0, 0, 0)
        text_size, _ = cv2.getTextSize(self.prompt, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(overlay, (x-10, y - text_h-10), (x + text_w + 10, y + 10), text_color_bg, -1)
        cv2.putText(overlay, self.prompt, pos, font, font_scale, text_color_fg, font_thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha,0, img)

        return img
    def parse_prompt(self):
        self.prompt = self.path.split("/")[-1].split("__")[0].replace("_", " ")
    def __eq__(self, other: object) -> bool:
        if isinstance(other, StableDiffusionImage):
            return self.path == other.path
        if isinstance(other, str):
            return self.path == other
        return False
    def __str__(self) -> str:
        return self.path
    def __repr__(self) -> str:
        return f"StableDiffusionImage(path={self.path}, score={self.score})"

class ImageQueue:
    def __init__(self):
        self.queue = []
    def enqueue(self, item):
        self.queue.append(item)
    def __getitem__(self, index):
        self._sort()
        return self.queue[index]
    def __len__(self):
        return len(self.queue)
    def _sort(self):
        self.queue.sort(key=lambda x: x.score + random() * 0.1, reverse=True)
        for entry in self.queue:
            entry.score += 0.1

image_queue = ImageQueue()


def fetch_images():
    files = os.listdir(IMAGE_PATH)
    for path in files:
        full_path = IMAGE_PATH + "/" + path
        if full_path not in image_queue.queue:
            image_queue.enqueue(StableDiffusionImage(full_path))

def next_image():
    fetch_images()
    image = image_queue[0]
    img = image.load()
    print(len(image_queue.queue))
    return img

def main():
    cv2.namedWindow("screen", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fetch_images()

    while 1:
        fetch_images()
        img = next_image()
        cv2.imshow("screen", img)

        if cv2.waitKey(SHOW_DURATION_MS) == 27:
            break

if __name__ == "__main__":
    main()

