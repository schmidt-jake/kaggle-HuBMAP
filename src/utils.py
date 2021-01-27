from typing import Dict, Tuple
import tensorflow as tf
import tensorflow_io as tfio


def rle2mask(mask_rle: tf.Tensor, width: tf.Tensor, height: tf.Tensor) -> tf.Tensor:
    # https://stackoverflow.com/a/58842473/14841071
    size = width * height
    s = tf.strings.split(mask_rle)
    s = tf.strings.to_number(s, tf.int64)
    starts = s[::2] - 1
    lens = s[1::2]
    total_ones = tf.reduce_sum(lens)  # Make ones to be scattered
    ones = tf.ones([total_ones], tf.uint8)
    r = tf.range(total_ones)  # Make scattering indices
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, "right")
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    return tf.reshape(mask_flat, (height, width))


def parse_image(
    x: Dict[str, tf.Tensor],
) -> Dict[str, tf.Tensor]:

    raw_bytes = tf.io.read_file(x["image_file"])
    x["image"] = tfio.experimental.image.decode_tiff(raw_bytes)
    x["mask"] = rle2mask(
        mask_rle=x["encoding"],
        width=x["width_pixels"],
        height=x["height_pixels"],
    )
    x["image_tiles"] = tile_image(x["image"])
    x["mask_tiles"] = tf.squeeze(tile_image(tf.expand_dims(x["mask"], axis=2)), axis=3)
    return x


def tile_image(
    x: tf.Tensor,
    size_rows: int = 1000,
    size_cols: int = 1000,
    stride_rows: int = 1000,
    stride_cols: int = 1000,
    rate_rows: int = 1,
    rate_cols: int = 1,
) -> tf.Tensor:
    x = tf.expand_dims(x, axis=0)
    tiles = tf.image.extract_patches(
        images=x,
        sizes=[1, size_rows, size_cols, 1],
        strides=[1, stride_rows, stride_cols, 1],
        rates=[1, rate_rows, rate_cols, 1],
        padding="VALID",
    )
    tiles = tf.squeeze(tiles, axis=0)
    # s = tf.shape(tiles)
    # tiles = tf.reshape(tiles, (s[0], s[1], size_rows, size_cols, tf.shape(x)[-1]))
    tiles = tf.reshape(tiles, (-1, size_rows, size_cols, tf.shape(x)[-1]))
    return tiles
