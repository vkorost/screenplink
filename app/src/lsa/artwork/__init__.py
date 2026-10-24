"""Image artwork for the figure drills: pure, Qt-free (cv2 and numpy only).

The CV core of this project stays free of Qt so it can survive a rewrite of the
shell. The seam is `lsa.modes.sprite`, the only place a grey array becomes a
QImage; everything in this package takes and returns numpy arrays and plain
numbers.
"""
