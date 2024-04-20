mod transformations_cache;

use crate::quadtree::Quadrant;
use image::{ImageBuffer, Luma};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::LeastSquaresSvdInto;
use rand::Rng;
use rayon::prelude::*;

pub(crate) type Float = f32;
pub(crate) type Index = u16;

const BRIGHTNESS_CLAMP: Float = 1000.0;
const CONTRAST_CLAMP: f32 = 2.5;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Header {
    pub(crate) len: u32,
    pub(crate) width: Index,
    pub(crate) height: Index,
}

impl Header {
    pub(crate) fn new(width: Index, height: Index, len: u32) -> Self {
        Self { width, height, len }
    }
}

#[derive(Debug)]
pub(crate) struct Transformations<'a> {
    pub(crate) header: Header,
    pub(crate) src_coordinate: &'a mut [Coordinate],
    pub(crate) dest_coordinate: &'a mut [Coordinate],
    pub(crate) scale: &'a mut [u8],
    pub(crate) is_flipped: &'a mut [u8],
    pub(crate) degrees: &'a mut [u8],
    pub(crate) adjustments: &'a mut [Adjustments],
    pub(crate) current: usize,
}

#[derive(Clone, Copy)]
struct Transformation {
    src_x: Index,
    src_y: Index,
    dest_x: Index,
    dest_y: Index,
    scale: u8,
    is_flipped: bool,
    degrees: Rotation,
    adjustments: Adjustments,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum Rotation {
    R0,
    R90,
    R180,
    R270,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Adjustments {
    brightness_raw: i8,
    contrast_raw: i8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coordinate {
    x: Index,
    y: Index,
}

impl Adjustments {
    fn new(brightness: Float, contrast: Float) -> Self {
        let brightness = brightness.clamp(-BRIGHTNESS_CLAMP, BRIGHTNESS_CLAMP);
        let contrast = contrast.clamp(-CONTRAST_CLAMP, CONTRAST_CLAMP);
        Self {
            brightness_raw: (brightness / BRIGHTNESS_CLAMP * 127.0).round() as i8,
            contrast_raw: (contrast / CONTRAST_CLAMP * 127.0).round() as i8,
        }
    }

    fn brightness(&self) -> Float {
        self.brightness_raw as Float / 127.0 * BRIGHTNESS_CLAMP
    }

    fn contrast(&self) -> Float {
        self.contrast_raw as Float / 127.0 * CONTRAST_CLAMP
    }
}

impl Default for Adjustments {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl Transformations<'_> {
    // fn len(&self) -> usize {
    //     self.width * self.height
    // }

    fn get_index(&self, i: usize) -> Transformation {
        let byte = self.is_flipped[i / 8];
        let is_flipped = (byte >> (i % 8)) & 1 == 1;

        let byte = self.degrees[i / 4];
        let degrees = match (byte >> ((i % 4) * 2)) & 3 {
            0 => Rotation::R0,
            1 => Rotation::R90,
            2 => Rotation::R180,
            3 => Rotation::R270,
            _ => unreachable!(),
        };

        Transformation {
            src_x: self.src_coordinate[i].x,
            src_y: self.src_coordinate[i].y,
            dest_x: self.dest_coordinate[i].x,
            dest_y: self.dest_coordinate[i].y,
            scale: self.scale[i],
            is_flipped,
            degrees,
            adjustments: self.adjustments[i],
        }
    }

    fn push(
        &mut self,
        Transformation {
            src_x,
            src_y,
            dest_x,
            dest_y,
            scale,
            is_flipped,
            degrees,
            adjustments,
        }: Transformation,
    ) {
        let i = self.current;
        self.src_coordinate[i] = Coordinate { x: src_x, y: src_y };
        self.dest_coordinate[i] = Coordinate {
            x: dest_x,
            y: dest_y,
        };
        self.scale[i] = scale;
        self.is_flipped[i / 8] |= (is_flipped as u8) << (i % 8);
        self.degrees[i / 4] |= (degrees as u8) << ((i % 4) * 2);
        self.adjustments[i] = adjustments;
        self.current += 1;
    }
}

pub(crate) fn compress(m: Array2<Float>, leaves: &[Quadrant], result: Transformations) {
    let mut cache = transformations_cache::Cache::default();
    for l in leaves {
        cache.get(&m, l.depth as u8);
    }

    let bar = indicatif::ProgressBar::new(leaves.len() as u64);
    let result_mutex = parking_lot::Mutex::new(result);
    let chunk_count = std::thread::available_parallelism().unwrap().get() * 3;
    leaves
        .par_chunks(leaves.len() / chunk_count)
        .for_each(|dest_quadrants| {
            for dest_quadrant in dest_quadrants {
                let mut min = Float::INFINITY;
                let mut min_t = None;
                let dest_block =
                    Array2::from_shape_fn((dest_quadrant.height, dest_quadrant.width), |(y, x)| {
                        m[[dest_quadrant.y + y, dest_quadrant.x + x]]
                    });
                let slot = cache.map.get(&(dest_quadrant.depth as u8)).unwrap();
                for (i, src_block) in slot.src_blocks.iter().enumerate() {
                    let adjustments = find_adjustments(src_block.clone(), dest_block.clone());
                    let s = src_block
                        .clone()
                        .map(|&x| x * adjustments.contrast() + adjustments.brightness());
                    let d =
                        ((dest_block.clone() - s.clone()) * (dest_block.clone() - s.clone())).sum();
                    if d < min {
                        let mut t = slot.transformations.get_index(i);
                        // skip if dest in src
                        if t.src_x as usize <= dest_quadrant.x
                            && t.src_y as usize <= dest_quadrant.y
                            && t.src_x as usize + t.scale as usize
                                >= dest_quadrant.x + dest_quadrant.width
                            && t.src_y as usize + t.scale as usize
                                >= dest_quadrant.y + dest_quadrant.height
                        {
                            continue;
                        }
                        t.adjustments = adjustments;
                        t.dest_x = dest_quadrant.x as u16;
                        t.dest_y = dest_quadrant.y as u16;
                        min_t = Some(t);
                        min = d;
                    }
                }
                result_mutex.lock().push(min_t.unwrap());
                bar.inc(1);
            }
        });
    bar.finish();

    // unsafe {
    //     std::alloc::dealloc(
    //         transformations.coordinate.as_mut_ptr().cast(),
    //         std::alloc::Layout::array::<Coordinate>(transformations.coordinate.len()).unwrap(),
    //     );
    //     std::alloc::dealloc(
    //         transformations.is_flipped.as_mut_ptr(),
    //         std::alloc::Layout::array::<u8>(transformations.is_flipped.len()).unwrap(),
    //     );
    //     std::alloc::dealloc(
    //         transformations.degrees.as_mut_ptr(),
    //         std::alloc::Layout::array::<u8>(transformations.degrees.len()).unwrap(),
    //     );
    //     std::alloc::dealloc(
    //         transformations.adjustments.as_mut_ptr().cast(),
    //         std::alloc::Layout::array::<Adjustments>(transformations.adjustments.len()).unwrap(),
    //     );
    // }
}

pub(crate) fn decompress(transformations: Transformations) -> Vec<Array2<Float>> {
    let mut rng = rand::thread_rng();
    let mut iterations = Vec::new();
    iterations.push(Array2::from_shape_fn(
        (
            transformations.header.height as usize,
            transformations.header.width as usize,
        ),
        |(_, _)| rng.gen_range(0..256) as Float,
    ));
    dbg!(&transformations.header);

    for i in 0..8 {
        let mut next = Array2::zeros((
            transformations.header.height as usize,
            transformations.header.width as usize,
        ));
        for t in 0..transformations.dest_coordinate.len() {
            let transformation = transformations.get_index(t);
            let src_width = transformations.header.width as usize
                / 2_usize.pow(transformation.scale as u32 - 1);
            let src_height = transformations.header.height as usize
                / 2_usize.pow(transformation.scale as u32 - 1);
            let src_block = reduce_block(
                {
                    let foo = iterations[i as usize].clone().slice_move(s![
                        transformation.src_y as usize..transformation.src_y as usize + src_height,
                        transformation.src_x as usize..transformation.src_x as usize + src_width,
                    ]);
                    Array2::from_shape_fn((src_height, src_width), |(y, x)| foo[[y, x]])
                },
                2,
            );
            let dest_block = transform(src_block, &transformation);
            let dest_width =
                transformations.header.width as usize / 2_usize.pow(transformation.scale as u32);
            let dest_height =
                transformations.header.height as usize / 2_usize.pow(transformation.scale as u32);
            next.slice_mut(s![
                transformation.dest_y as usize..transformation.dest_y as usize + dest_height,
                transformation.dest_x as usize..transformation.dest_x as usize + dest_width,
            ])
            .assign(&dest_block);
        }
        iterations.push(next);
    }

    iterations
}

fn find_adjustments_simple(src: Array2<Float>, dest: Array2<Float>) -> Adjustments {
    let contrast = 0.75;
    let temp = &dest - contrast * src;
    let brightness = temp.sum() as Float / dest.len() as Float;
    Adjustments::new(brightness, contrast)
}

fn find_adjustments(src: Array2<Float>, dest: Array2<Float>) -> Adjustments {
    let ones: Array1<Float> = Array1::ones(src.len());
    let src: Array1<Float> = src.clone().into_shape(src.len()).unwrap();
    let a: Array2<Float> = ndarray::stack(Axis(1), &[ones.view(), src.view()]).unwrap();
    let b: Array1<Float> = dest.clone().into_shape(dest.len()).unwrap();
    let x = a.least_squares_into(b).unwrap();

    Adjustments::new(x.solution[0], x.solution[1])
}

pub(crate) fn reduce(img: &ImageBuffer<Luma<u8>, Vec<u8>>, factor: usize) -> Array2<Float> {
    Array2::from_shape_fn(
        (
            img.height() as usize / factor,
            img.width() as usize / factor,
        ),
        |(y, x)| avg_pixel(img, x * factor, y * factor, factor),
    )
}

fn reduce_block(m: Array2<Float>, factor: usize) -> Array2<Float> {
    Array2::from_shape_fn((m.nrows() / factor, m.ncols() / factor), |(y, x)| {
        let mut sum = 0.0;
        for i in 0..factor {
            for j in 0..factor {
                sum += m[[y * factor + i, x * factor + j]];
            }
        }
        sum / (factor * factor) as Float
    })
}

fn avg_pixel(img: &ImageBuffer<Luma<u8>, Vec<u8>>, x: usize, y: usize, factor: usize) -> Float {
    let mut sum = 0.0;
    for i in 0..factor {
        for j in 0..factor {
            sum += img.get_pixel((x + i) as u32, (y + j) as u32).0[0] as Float;
        }
    }
    sum / (factor * factor) as Float
}

fn rotate(mut m: Array2<Float>, degrees: Rotation) -> Array2<Float> {
    match degrees {
        Rotation::R0 => {}
        Rotation::R90 => {
            m = Array2::from_shape_fn((m.nrows(), m.ncols()), |(y, x)| m[[x, m.ncols() - y - 1]]);
        }
        Rotation::R180 => {
            m = Array2::from_shape_fn((m.nrows(), m.ncols()), |(y, x)| {
                m[[m.nrows() - y - 1, m.ncols() - x - 1]]
            });
        }
        Rotation::R270 => {
            m = Array2::from_shape_fn((m.nrows(), m.ncols()), |(y, x)| m[[m.nrows() - x - 1, y]]);
        }
    };

    m
}

fn flip(m: Array2<Float>, flip: bool) -> Array2<Float> {
    if flip {
        return Array2::from_shape_fn((m.nrows(), m.ncols()), |(y, x)| m[[m.nrows() - y - 1, x]]);
    }

    m
}

fn transform(m: Array2<Float>, transformation: &Transformation) -> Array2<Float> {
    let Transformation {
        is_flipped,
        degrees,
        adjustments,
        ..
    } = *transformation;

    adjustments.contrast() * rotate(flip(m, is_flipped), degrees) + adjustments.brightness()
}
