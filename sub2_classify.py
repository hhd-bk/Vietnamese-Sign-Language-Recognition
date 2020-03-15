import numpy as np
from PIL import ImageDraw, ImageFont, Image

# Load unicode font
unicode_font = ImageFont.truetype("font-times-new-roman.ttf", 20)

# Put the unicode text to the screen
def put_text_onscreen(frame, unicode_text, x_pos, y_pos):
    # Load img into PIL format
    frame = Image.fromarray(frame)
    # Draw text to image
    draw = ImageDraw.Draw(frame)
    draw.text((x_pos, y_pos), unicode_text, font=unicode_font, fill=(255, 255, 255))
    # Convert to image to print screen
    frame = np.array(frame)
    return frame

def update_sequence(sequence, sequence1, c1):
    return