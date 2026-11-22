#!/usr/bin/env python3
"""real_model_demo.py <out.svg> [out.json]: run the repo's EmotionClassifier (fer: Haar face detector +
FER2013 CNN) on scikit-image's astronaut photo (NASA, public domain) and draw the per-emotion scores and
the decision EmotionClassifier.decide makes from them. Needs fer, tensorflow-cpu, scikit-image."""
import json
import sys

import cv2
from skimage import data

from facial_emotion_detector.emotion_classifier import EmotionClassifier

out_svg = sys.argv[1]
frame = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
clf = EmotionClassifier()
faces = clf.detector.detect_emotions(frame)
if not faces:
    sys.exit('no face detected')
scores = {k: float(v) for k, v in faces[0]['emotions'].items()}
decision = clf.decide(scores)
result = {'faces': len(faces), 'box': [int(v) for v in faces[0]['box']], 'scores': scores, 'decision': decision.name}
print(json.dumps(result))
if len(sys.argv) > 2:
    json.dump(result, open(sys.argv[2], 'w'), indent=1)

W, L, BW, RH = 520, 90, 340, 26
rows = sorted(scores.items(), key=lambda kv: -kv[1])
H = 60 + RH * len(rows) + 40
el = [f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
      f'<text x="12" y="22" fill="#10161e" font-weight="700">fer scores on one test photo, and the node\'s decision</text>']
for i, (name, v) in enumerate(rows):
    y = 44 + i * RH
    el.append(f'<text x="{L - 8}" y="{y + 15}" text-anchor="end" fill="#3f4d58">{name}</text>')
    el.append(f'<rect x="{L}" y="{y + 3}" width="{BW}" height="16" fill="#f1f4f6"/>')
    el.append(f'<rect x="{L}" y="{y + 3}" width="{BW * v:.1f}" height="16" fill="#005f73"/>')
    el.append(f'<text x="{L + BW + 8}" y="{y + 15}" fill="#3f4d58">{v:.2f}</text>')
el.append(f'<text x="12" y="{H - 26}" fill="#10161e">decision (EmotionClassifier.decide): {decision.name}</text>')
el.append(f'<text x="12" y="{H - 8}" fill="#6b7785" font-size="11">Input: scikit-image data.astronaut() (NASA, public domain), '
          f'one face found. From CI, not a webcam run.</text>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, sans-serif" font-size="12">\n' + '\n'.join(el) + '\n</svg>\n')
open(out_svg, 'w').write(svg)
