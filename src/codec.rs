use image::{ImageBuffer, Luma};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::LeastSquaresSvdInto;
use rand::Rng;

type Float = f32;
type Index = u16;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Adjustments {
    brightness: Float,
    contrast: Float,
}

impl Default for Adjustments {
    fn default() -> Self {
        Adjustments {
            brightness: 0.0,
            contrast: 1.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Transformations<'a> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) coordinate: &'a mut [Coordinate],
    pub(crate) is_flipped: &'a mut [u8],
    pub(crate) degrees: &'a mut [u8],
    pub(crate) adjustments: &'a mut [Adjustments],
    pub(crate) current: usize,
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
            x: self.coordinate[i].x,
            y: self.coordinate[i].y,
            is_flipped,
            degrees,
            adjustments: self.adjustments[i],
        }
    }

    fn get(&self, x: usize, y: usize) -> Transformation {
        self.get_index(y * self.width + x)
    }

    fn push(
        &mut self,
        Transformation {
            x,
            y,
            is_flipped,
            degrees,
            adjustments,
        }: Transformation,
    ) {
        let i = self.current;

        self.coordinate[i] = Coordinate { x, y };
        self.is_flipped[i / 8] |= (is_flipped as u8) << (i % 8);
        self.degrees[i / 4] |= (degrees as u8) << ((i % 4) * 2);
        self.adjustments[i] = adjustments;

        self.current += 1;
    }
}

#[derive(Clone, Copy)]
struct Transformation {
    x: Index,
    y: Index,
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
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coordinate {
    x: Index,
    y: Index,
}

pub(crate) fn compress(
    m: Array2<Float>,
    src_size: usize,
    dest_size: usize,
    step: usize,
    mut result: Transformations,
) {
    let (transformations, src_blocks) =
        gen_all_transformations(m.clone(), src_size, dest_size, step);

    let width = m.ncols() / dest_size;
    let height = m.nrows() / dest_size;
    let bar = indicatif::ProgressBar::new((width * height) as u64);
    for y in 0..(height) {
        for x in 0..(width) {
            let mut min = Float::INFINITY;
            let mut min_t = None;
            let dest_block = m.clone().slice_move(s![
                y * dest_size..(y + 1) * dest_size,
                x * dest_size..(x + 1) * dest_size
            ]);
            let dest_block =
                Array2::from_shape_fn((dest_size, dest_size), |(y, x)| dest_block[[y, x]]);
            for (i, src_block) in src_blocks.iter().enumerate() {
                let adjustments = find_adjustments(src_block.clone(), dest_block.clone());
                transformations.adjustments[i] = adjustments;
                let s = src_block
                    .clone()
                    .map(|&x| x * adjustments.contrast as Float + adjustments.brightness as Float);
                let d = ((dest_block.clone() - s.clone()) * (dest_block.clone() - s.clone())).sum();
                if d < min {
                    min_t = Some(transformations.get_index(i));
                    min = d;
                }
            }
            result.push(min_t.unwrap());
            bar.inc(1);
        }
    }
    bar.finish();

    unsafe {
        std::alloc::dealloc(
            transformations.coordinate.as_mut_ptr().cast(),
            std::alloc::Layout::array::<Coordinate>(transformations.coordinate.len()).unwrap(),
        );
        std::alloc::dealloc(
            transformations.is_flipped.as_mut_ptr(),
            std::alloc::Layout::array::<u8>(transformations.is_flipped.len()).unwrap(),
        );
        std::alloc::dealloc(
            transformations.degrees.as_mut_ptr(),
            std::alloc::Layout::array::<u8>(transformations.degrees.len()).unwrap(),
        );
        std::alloc::dealloc(
            transformations.adjustments.as_mut_ptr().cast(),
            std::alloc::Layout::array::<Adjustments>(transformations.adjustments.len()).unwrap(),
        );
    }
}

pub(crate) fn decompress(
    transformations: Transformations,
    src_size: usize,
    dest_size: usize,
    step: usize,
) -> Vec<Array2<Float>> {
    let mut rng = rand::thread_rng();
    let factor = src_size / dest_size;
    let mut iterations = Vec::new();
    iterations.push(Array2::from_shape_fn(
        (
            transformations.height * dest_size,
            transformations.width * dest_size,
        ),
        |(_, _)| rng.gen_range(0..256) as Float,
    ));

    for i in 0..8 {
        let mut next = Array2::zeros((
            transformations.height * dest_size,
            transformations.width * dest_size,
        ));
        for y in 0..transformations.height {
            for x in 0..transformations.width {
                let transformation = transformations.get(x, y);
                let src_block = reduce_block(
                    {
                        let foo = iterations[i].clone().slice_move(s![
                            transformation.y as usize * step
                                ..(transformation.y as usize + 1) * src_size,
                            transformation.x as usize * step
                                ..(transformation.x as usize + 1) * src_size,
                        ]);
                        Array2::from_shape_fn((src_size, src_size), |(y, x)| foo[[y, x]])
                    },
                    factor,
                );
                let dest_block = transform(src_block, &transformation);
                next.slice_mut(s![
                    y * dest_size..(y + 1) * dest_size,
                    x * dest_size..(x + 1) * dest_size,
                ])
                .assign(&dest_block);
            }
        }
        iterations.push(next);
    }

    iterations
}

fn find_adjustments(src: Array2<Float>, dest: Array2<Float>) -> Adjustments {
    let ones: Array1<Float> = Array1::ones(src.len());
    let src: Array1<Float> = src.clone().into_shape(src.len()).unwrap();
    let a: Array2<Float> = ndarray::stack(Axis(1), &[ones.view(), src.view()]).unwrap();
    let b: Array1<Float> = dest.clone().into_shape(dest.len()).unwrap();
    let x = a.least_squares_into(b).unwrap();

    Adjustments {
        contrast: x.solution[1],
        brightness: x.solution[0],
    }
}

fn gen_all_transformations(
    m: Array2<Float>,
    src_size: usize,
    dest_size: usize,
    step: usize,
) -> (Transformations<'static>, Vec<Array2<Float>>) {
    let factor = src_size / dest_size;
    let mut blocks = Vec::new();

    let height = (m.nrows() - src_size) / step + 1;
    let width = (m.ncols() - src_size) / step + 1;
    let mut transformations: Transformations<'static> = unsafe {
        Transformations {
            width,
            height,
            coordinate: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::array::<Coordinate>(width * height * 2 * 4).unwrap(),
                )
                .cast(),
                width * height * 2 * 4,
            ),
            is_flipped: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::array::<u8>((width * height * 2 * 4).div_ceil(8)).unwrap(),
                ),
                (width * height * 2 * 4).div_ceil(8),
            ),
            degrees: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::array::<u8>((width * height * 2 * 4).div_ceil(4)).unwrap(),
                ),
                (width * height * 2 * 4).div_ceil(4),
            ),
            adjustments: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(
                    std::alloc::Layout::array::<Adjustments>(width * height * 2 * 4).unwrap(),
                )
                .cast(),
                width * height * 2 * 4,
            ),
            current: 0,
        }
    };

    for y in 0..height {
        for x in 0..width {
            let src_block = reduce_block(
                m.clone().slice_move(s![
                    y * step..y * step + src_size,
                    x * step..x * step + src_size
                ]),
                factor,
            );
            for is_flipped in [false, true] {
                for degrees in [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270] {
                    let t = Transformation {
                        x: x as Index,
                        y: y as Index,
                        is_flipped,
                        degrees,
                        adjustments: Adjustments::default(),
                    };
                    blocks.push(transform(src_block.clone(), &t));
                    transformations.push(t);
                }
            }
        }
    }

    (transformations, blocks)
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

    adjustments.contrast * rotate(flip(m, is_flipped), degrees) + adjustments.brightness
}
