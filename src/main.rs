use image::{ImageBuffer, Luma};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::LeastSquaresSvdInto;
use rand::Rng;

type Float = f32;
type Index = u16;

#[derive(Clone, Copy)]
struct Adjustments {
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

struct Transformations {
    width: usize,
    height: usize,
    transformations: Vec<Transformation>,
}

#[derive(Clone, Copy)]
struct Transformation {
    x: Index,
    y: Index,
    is_flipped: bool,
    degrees: Rotation,
    adjustments: Adjustments,
}

#[derive(Clone, Copy)]
enum Rotation {
    R0,
    R90,
    R180,
    R270,
}

struct TransformedBlock {
    block: Array2<Float>,
    transformation: Transformation,
}

fn main() {
    // Ensure image is grayscale
    let img = image::open("monkey.gif").unwrap().to_luma8();
    // Crash if image is not square
    // assert_eq!(img.width(), img.height());

    // Reduce and save each rotation
    let t = compress(reduce(&img, 4), 8, 4, 8);

    dbg!(t.transformations.len() * std::mem::size_of::<Transformation>());

    let iterations = decompress(t, 8, 4, 8);

    for (i, iteration) in iterations.iter().enumerate() {
        let img = ImageBuffer::from_fn(
            iteration.nrows() as u32,
            iteration.ncols() as u32,
            |x, y| Luma([iteration[[y as usize, x as usize]] as u8]),
        );
        img.save(format!("output-{}.jpg", i)).unwrap();
    }
}

fn compress(m: Array2<Float>, src_size: usize, dest_size: usize, step: usize) -> Transformations {
    let mut transformations = Vec::new();
    let transformed_blocks = gen_all_transformations(m.clone(), src_size, dest_size, step);

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
            for TransformedBlock {
                block: src_block,
                mut transformation,
            } in &transformed_blocks
            {
                let adjustments = find_adjustments(src_block.clone(), dest_block.clone());
                transformation.adjustments = adjustments;
                let s = src_block
                    .clone()
                    .map(|&x| x * adjustments.contrast as Float + adjustments.brightness as Float);
                let d = ((dest_block.clone() - s.clone()) * (dest_block.clone() - s.clone())).sum();
                if d < min {
                    min_t = Some(transformation);
                    min = d;
                }
            }
            transformations.push(min_t.unwrap());
            bar.inc(1);
        }
    }

    Transformations {
        width,
        height,
        transformations,
    }
}

fn decompress(
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
                let transformation = transformations.transformations[y * transformations.width + x];
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
) -> Vec<TransformedBlock> {
    let factor = src_size / dest_size;
    let mut res = Vec::new();

    for y in 0..((m.nrows() - src_size) / step + 1) {
        for x in 0..((m.ncols() - src_size) / step + 1) {
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
                    res.push(TransformedBlock {
                        block: transform(src_block.clone(), &t),
                        transformation: t,
                    });
                }
            }
        }
    }

    res
}

fn reduce(img: &ImageBuffer<Luma<u8>, Vec<u8>>, factor: usize) -> Array2<Float> {
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
