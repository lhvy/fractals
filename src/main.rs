use clap::Parser;
mod cli;
mod codec;
mod quadlist;
mod quadtree;

use cli::Args;
use codec::{Adjustments, Coordinate, Header, Transformations};
use image::{ImageBuffer, Luma};
use lzma::EXTREME_PRESET;
use mimalloc::MiMalloc;
use quadtree::Quadrant;
use std::io::{Read, Write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const FACTOR: usize = 1;

fn main() {
    let args = Args::parse();

    match args.cmd {
        cli::Command::Encode(e) => {
            compress_command(e);
        }
        cli::Command::Decode(d) => {
            let mut compressed = std::fs::File::open(d.file).unwrap();
            decompress(&mut compressed, d.output);
        }
        cli::Command::Test(t) => {
            compress_command(t.clone());
            let path = match &t.output {
                Some(output) => output.to_string() + "/compressed.leic",
                None => "output/compressed.leic".to_string(),
            };
            let mut compressed = std::fs::File::open(path).unwrap();
            decompress(&mut compressed, t.output);
        }
        cli::Command::Error(e) => {
            let input = image::open(e.input).unwrap().to_luma8();
            let output = image::open(e.output).unwrap().to_luma8();
            compare_error(input, output);
        }
    };
}

fn compress_command(e: cli::Encode) {
    println!("Generating quadtree");
    // Open image as RGB
    let img = image::open(e.file.clone()).unwrap().to_rgb8();
    let storage = quadtree::Storage::new(img, e.max_depth, e.detail_threshold);
    let quadtree = quadtree::Quadtree::new(&storage);
    let mut leaves = quadtree.leaves();

    // Sort leaves by leave.index
    leaves.sort_by_key(|leave| leave.index);
    // Indexes are increasing, but make them consecutive starting from 0
    for (i, leave) in leaves.iter_mut().enumerate() {
        leave.index = i;
    }

    // Ensure image is grayscale
    let img = image::open(e.file).unwrap().to_luma8();
    // Crash if image is not square
    assert_eq!(img.width(), img.height());

    let path = match e.output {
        Some(output) => output,
        None => "output".to_string(),
    };
    let _ = std::fs::create_dir(&path);
    let compressed = std::fs::File::create(format!("{}/compressed.leic", path)).unwrap();
    quadtree
        .create_image(&storage)
        .save(format!("{}/quadtree.jpg", path))
        .unwrap();
    compress(img, &leaves, &compressed);
    drop(compressed);
}

fn compress(
    img: ImageBuffer<Luma<u8>, Vec<u8>>,
    leaves: &[Quadrant],
    mut compressed: &std::fs::File,
) {
    let len = leaves.len();
    let width = img.width() as usize / FACTOR;
    let height = img.height() as usize / FACTOR;
    let header = Header::new(width as u16, height as u16, len as u32);

    let header_size = std::mem::size_of::<Header>();
    let src_coordinate_size = std::mem::size_of::<Coordinate>() * len;
    let scale_size = len;
    let is_flipped_size = len.div_ceil(8);
    let degrees_size = len.div_ceil(4);
    let adjustments_size = std::mem::size_of::<Adjustments>() * len;
    let file_len = header_size
        + src_coordinate_size
        + scale_size
        + is_flipped_size
        + degrees_size
        + adjustments_size;

    let data = unsafe {
        std::slice::from_raw_parts_mut(
            std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(file_len, 4).unwrap()),
            file_len,
        )
    };

    let transformations = unsafe {
        let ptr = data.as_mut_ptr().cast::<Header>();
        *ptr = header;

        let src_coordinate = ptr.add(1).cast::<u8>();
        let scale = src_coordinate.add(src_coordinate_size);
        let adjustments = scale.add(scale_size);
        let is_flipped = adjustments.add(adjustments_size);
        let degrees = is_flipped.add(is_flipped_size);

        Transformations {
            header,
            src_coordinate: std::slice::from_raw_parts_mut(src_coordinate.cast(), len),
            scale: std::slice::from_raw_parts_mut(scale.cast(), len),
            is_flipped: std::slice::from_raw_parts_mut(is_flipped, is_flipped_size),
            degrees: std::slice::from_raw_parts_mut(degrees, degrees_size),
            adjustments: std::slice::from_raw_parts_mut(adjustments.cast(), len),
            current: 0,
        }
    };

    println!("Applying fractal compression");
    codec::compress(codec::reduce(&img, FACTOR), leaves, transformations);

    println!(
        "       fractal compressed file with size: {:>9}",
        data.len()
    );
    compressed
        .write_all(lzma::compress(data, 9 | EXTREME_PRESET).unwrap().as_slice())
        .unwrap();
    println!(
        "lzma + fractal compressed file with size: {:>9}",
        compressed.metadata().unwrap().len()
    );

    unsafe {
        std::alloc::dealloc(
            data.as_mut_ptr(),
            std::alloc::Layout::from_size_align(file_len, 8).unwrap(),
        );
    }
}

fn decompress(compressed: &mut std::fs::File, output: Option<String>) {
    println!("Decompressing lzma file");
    let mut data = Vec::new();
    compressed.read_to_end(&mut data).unwrap();
    let mut data = lzma::decompress(&data).unwrap();

    println!("Decompressing fractal file");
    let t = unsafe {
        let ptr = data.as_mut_ptr().cast::<Header>();
        let header = *ptr;
        let len = header.len as usize;

        let src_coordinate_size = len * std::mem::size_of::<Coordinate>();
        let scale_size = len;
        let is_flipped_size = len.div_ceil(8);
        let degrees_size = len.div_ceil(4);
        let adjustments_size = std::mem::size_of::<Adjustments>() * len;

        let src_coordinate = ptr.add(1).cast::<u8>();
        let scale = src_coordinate.add(src_coordinate_size);
        let adjustments = scale.add(scale_size);
        let is_flipped = adjustments.add(adjustments_size);
        let degrees = is_flipped.add(is_flipped_size);

        Transformations {
            header,
            src_coordinate: std::slice::from_raw_parts_mut(src_coordinate.cast(), len),
            scale: std::slice::from_raw_parts_mut(scale.cast(), len),
            is_flipped: std::slice::from_raw_parts_mut(is_flipped, is_flipped_size),
            degrees: std::slice::from_raw_parts_mut(degrees, degrees_size),
            adjustments: std::slice::from_raw_parts_mut(adjustments.cast(), len),
            current: 0,
        }
    };

    let iterations = codec::decompress(t);

    let folder_name = match output {
        Some(output) => output,
        None => "output".to_string(),
    };
    let _ = std::fs::create_dir(&folder_name);
    for (i, iteration) in iterations.iter().enumerate() {
        let img = ImageBuffer::from_fn(
            iteration.nrows() as u32,
            iteration.ncols() as u32,
            |x, y| Luma([iteration[[y as usize, x as usize]] as u8]),
        );

        img.save(format!("{}/output-{}.jpg", folder_name, i))
            .unwrap();
    }
}

fn compare_error(input: ImageBuffer<Luma<u8>, Vec<u8>>, output: ImageBuffer<Luma<u8>, Vec<u8>>) {
    // Assert images are the same size
    assert_eq!(input.width(), output.width());
    assert_eq!(input.height(), output.height());

    // Peak Signal to Noise Ratio
    let sq = input
        .pixels()
        .zip(output.pixels())
        .map(|(i, o)| {
            let i = i[0] as f64;
            let o = o[0] as f64;
            (i - o).powi(2)
        })
        .sum::<f64>();
    let mse = sq / (input.width() * input.height()) as f64;
    let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
    println!("PSNR: {}", psnr);
}
