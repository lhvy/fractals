mod cli;
mod codec;

use clap::Parser;
use cli::Args;
use codec::{Adjustments, Coordinate, Transformations};
use image::{ImageBuffer, Luma};
use lzma::EXTREME_PRESET;
use mimalloc::MiMalloc;
use std::io::{Read, Write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const FACTOR: usize = 1;

fn main() {
    let args = Args::parse();

    let dest_size = match &args.cmd {
        cli::Command::Encode(e) => e.dest_size,
        cli::Command::Decode(d) => d.dest_size,
        cli::Command::Test(t) => t.dest_size,
        cli::Command::Error(_) => 4,
    };
    // if dest_size is not power of 2, complain
    if dest_size & (dest_size - 1) != 0 {
        eprintln!("dest_size must be power of 2");
        std::process::exit(1);
    }
    let src_size: usize = dest_size * 2;

    match args.cmd {
        cli::Command::Encode(e) => {
            compress_command(e, src_size, dest_size);
        }
        cli::Command::Decode(d) => {
            let path = match d.output {
                Some(path) => path,
                None => "output".to_string(),
            };
            let mut compressed = std::fs::File::open(d.file).unwrap();
            decompress(&mut compressed, &path, src_size, dest_size);
        }
        cli::Command::Test(t) => {
            compress_command(t.clone(), src_size, dest_size);
            let path = match t.output {
                Some(path) => path,
                None => "output".to_string(),
            };
            let mut compressed = std::fs::File::open(format!("{}/compressed.leic", path)).unwrap();
            decompress(&mut compressed, &path, src_size, dest_size);
        }
        cli::Command::Error(e) => {
            let input = image::open(e.input).unwrap().to_luma8();
            let output = image::open(e.output).unwrap().to_luma8();
            compare_error(input, output);
        }
    }
}

fn compress_command(value: cli::Value, src_size: usize, dest_size: usize) {
    // Ensure image is grayscale
    let img = image::open(value.file).unwrap().to_luma8();
    // Crash if image is not square
    assert_eq!(img.width(), img.height());

    let path = match value.output {
        Some(path) => path,
        None => "output".to_string(),
    };
    let _ = std::fs::create_dir(&path);

    let compressed = std::fs::File::create(format!("{}/compressed.leic", path)).unwrap();
    compress(img, &compressed, src_size, dest_size);
    drop(compressed);
}

fn compress(
    img: ImageBuffer<Luma<u8>, Vec<u8>>,
    mut compressed: &std::fs::File,
    src_size: usize,
    dest_size: usize,
) {
    let width = img.width() as usize / FACTOR / dest_size;
    let height = img.height() as usize / FACTOR / dest_size;
    let len = width * height;

    let header_size = std::mem::size_of::<usize>() * 2;
    let coordinate_size = std::mem::size_of::<Coordinate>() * len;
    let is_flipped_size = len.div_ceil(8);
    let degrees_size = len.div_ceil(4);
    let adjustments_size = std::mem::size_of::<Adjustments>() * len;
    let file_len =
        header_size + coordinate_size + is_flipped_size + degrees_size + adjustments_size;

    let data = unsafe {
        std::slice::from_raw_parts_mut(
            std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(file_len, 8).unwrap()),
            file_len,
        )
    };

    let transformations = unsafe {
        let mut ptr = data.as_mut_ptr().cast::<usize>();
        *ptr = width;
        ptr = ptr.add(1);
        *ptr = height;
        ptr = ptr.add(1);

        let coordinate = ptr.cast::<u8>();
        let adjustments = coordinate.add(coordinate_size);
        let is_flipped = adjustments.add(adjustments_size);
        let degrees = is_flipped.add(is_flipped_size);

        Transformations {
            width,
            height,
            coordinate: std::slice::from_raw_parts_mut(coordinate.cast(), len),
            is_flipped: std::slice::from_raw_parts_mut(is_flipped, is_flipped_size),
            degrees: std::slice::from_raw_parts_mut(degrees, degrees_size),
            adjustments: std::slice::from_raw_parts_mut(adjustments.cast(), len),
            current: 0,
        }
    };

    codec::compress(
        codec::reduce(&img, FACTOR),
        src_size,
        dest_size,
        transformations,
    );

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

fn decompress(compressed: &mut std::fs::File, path: &str, src_size: usize, dest_size: usize) {
    println!("Decompressing lzma file");
    let mut data = Vec::new();
    compressed.read_to_end(&mut data).unwrap();
    let mut data = lzma::decompress(&data).unwrap();

    println!("Decompressing fractal file");
    let t = unsafe {
        let mut ptr = data.as_mut_ptr().cast::<usize>();
        let width = *ptr;
        ptr = ptr.add(1);
        let height = *ptr;
        ptr = ptr.add(1);

        let len = width * height;
        let coordinate_size = std::mem::size_of::<Coordinate>() * len;
        let is_flipped_size = len.div_ceil(8);
        let degrees_size = len.div_ceil(4);
        let adjustments_size = std::mem::size_of::<Adjustments>() * len;

        let coordinate = ptr.cast::<u8>();
        let adjustments = coordinate.add(coordinate_size);
        let is_flipped = adjustments.add(adjustments_size);
        let degrees = is_flipped.add(is_flipped_size);

        Transformations {
            width,
            height,
            coordinate: std::slice::from_raw_parts_mut(coordinate.cast(), len),
            is_flipped: std::slice::from_raw_parts_mut(is_flipped, is_flipped_size),
            degrees: std::slice::from_raw_parts_mut(degrees, degrees_size),
            adjustments: std::slice::from_raw_parts_mut(adjustments.cast(), len),
            current: 0,
        }
    };

    let iterations = codec::decompress(t, src_size, dest_size);

    let _ = std::fs::create_dir(path);
    for (i, iteration) in iterations.iter().enumerate() {
        let img = ImageBuffer::from_fn(
            iteration.nrows() as u32,
            iteration.ncols() as u32,
            |x, y| Luma([iteration[[y as usize, x as usize]] as u8]),
        );
        img.save(format!("{}/output-{}.jpg", path, i)).unwrap();
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
