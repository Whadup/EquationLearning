def convert_to_la(img):
    return img.convert(mode='LA')


def take_alpha_channel(img):
    return img[1]

