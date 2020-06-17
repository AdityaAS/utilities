import cv2
import types
import numbers
import numpy as np
import imgaug as ia
import torch, torchvision
from torchvision import transforms
from PIL import Image, PILLOW_VERSION
from imgaug import augmenters as iaa
from utils.utils import reflect_lsp_kp, reflect_lsp_skp, translate_kps

def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
    #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    import math

    angle = math.radians(angle)
    if isinstance(shear, (tuple, list)) and len(shear) == 2:
        shear = [math.radians(s) for s in shear]
    elif isinstance(shear, numbers.Number):
        shear = math.radians(shear)
        shear = [shear, 0]
    else:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
        -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix

def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """
    Apply affine transformation on the image keeping image center invariant
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
        If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
        the second value corresponds to a shear parallel to the y axis.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)

class RandomAffine(object):
    """
    Random affine transformation of the image keeping center invariant
    Args:
        translate (int, optional): int of maximum absolute fraction for horizontal
            and vertical translations. For example translate=a, then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * a < dy < img_height * a.
    """

    def __init__(self, translate):
        if not (0.0 <= translate <= 1.0):
            raise ValueError("translation value should be between 0 and 1")

        self.translate = translate

    @staticmethod
    def get_params(translate, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        max_dx = translate * img_size[0]
        max_dy = translate * img_size[1]
        translations = (int(np.round(np.random.uniform(-max_dx, max_dx))),
                        int(np.round(np.random.uniform(-max_dy, max_dy))))

        angle = 0
        scale = 1.0
        shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, kps=None, skps=None):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        shape = img.shape[:2][::-1]
        ret = self.get_params(self.translate, shape)
        offset = ret[1]

        # img = translate_image(img, offset[0], axis=0)
        # img = translate_image(img, offset[1], axis=1)
        img = affine(Image.fromarray(img), *ret)

        kps = translate_kps(kps, offset, shape)
        skps = translate_kps(skps, offset, shape)

        return np.array(img).astype(np.uint8), kps, skps

    def __repr__(self):
        s = '(translate={translate}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kps=None, skps=None):
        for t in self.transforms:
            img, kps, skps = t(img, kps, skps)
        return img, kps, skps

class HorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, kps=None, skps=None):
        if np.random.random() <= self.flip_prob:
            img = cv2.flip(img, 0)
            h, w = img.shape[:2]

            if kps is not None:
                kps = reflect_lsp_kp(kps, size=h, axis=1)
            if skps is not None:
                skps = reflect_lsp_skp(skps, size=h, axis=1)

        return img, kps, skps

class VerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, kps=None, skps=None):
        if np.random.random() <= self.flip_prob:
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            if kps is not None:
                kps = reflect_lsp_kp(kps, size=w, axis=0)
            if skps is not None:
                skps = reflect_lsp_skp(skps, size=w, axis=0)

        return img, kps, skps

class TranslateHorizontal(object):
    def __init__(self, offset):
        self.offset = np.random.randint(0, offset)

    def __call__(self, img, kps=None, skps=None):
        translated_image = np.zeros(img.shape)
        translated_image[:, self.offset:, :] = img[:, :-self.offset, :]

        if kps is not None:

            kps[:, 0] = self.offset + kps[:, 0]
        if skps is not None:
            skps[:, 0] = self.offset + skps[:, 0]
        return translated_image, kps, skps

class TranslateVertical(object):
    def __init__(self, offset):
        self.offset = np.random.randint(0, offset)

    def __call__(self, img, kps=None, skps=None):
        translated_image = np.zeros(img.shape)
        translated_image[self.offset:, :, :] = img[:-self.offset, :, :]

        if kps is not None:
            kps[:, 1] = self.offset + kps[:, 1]
        if skps is not None:
            skps[:, 1] = self.offset + skps[:, 1]

        return translated_image, kps, skps

class MathAndBlur(object):
    def __init__(self, contrast_range=(0.5, 2.0), add_limit=25,
                 multiply_range=(0.75, 1.25), sigma_range=(0, 3.0),
                 k=5):
        self.contrast_range = contrast_range
        self.add_limit = add_limit
        self.multiply_range = multiply_range
        self.sigma_range = sigma_range
        self.k = k

    def __call__(self, image, kps, skps):
        im = image.copy()
        if np.random.randint(2):

            seq = iaa.Sequential(
                [
                    iaa.SomeOf((1, 4),
                            [
                                iaa.ContrastNormalization(self.contrast_range),
                                iaa.Add((-self.add_limit, self.add_limit)),
                                iaa.Multiply(self.multiply_range),
                                iaa.GaussianBlur(self.sigma_range),
                                iaa.MotionBlur(self.k)
                            ],
                            random_order=True)
                ])

            im = seq.augment_image(im)

        return im, kps, skps


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)


    def __call__(self, img, kps=None, skps=None):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        return np.array(self.transform(Image.fromarray(img))), kps, skps



class Augmentation(object):
    def __init__(self, blur, flip, flip_prob, translate, color_jitter, brightness=0.1, contrast=0.1):
        augmentations = []

        if blur:
            augmentations.append(MathAndBlur())

        if flip:
            augmentations.append(HorizontalFlip(flip_prob))
            augmentations.append(VerticalFlip(flip_prob))

        if translate:
            augmentations.append(RandomAffine(translate))

        if color_jitter:
            augmentations.append(ColorJitter(brightness, contrast))

        self.augment = Compose(augmentations)

    def __call__(self, img, kps=None, skps=None):
        return self.augment(img, kps, skps)




if __name__ == '__main__':
    transform = HMRAugmentation(True, True, 1.0, 0.1, True)

    img = np.random.random((224, 224, 3)).astype(np.uint8)
    kps = np.random.random((14, 3))
    skps = np.random.random((10, 3))

    t_img, t_kps, t_skps = transform(img, kps, skps)
    import pdb; pdb.set_trace()


