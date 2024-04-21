mod codec;

use codec::{Adjustments, Coordinate, Transformations};
use image::{ImageBuffer, Luma};
use lzma::EXTREME_PRESET;
use mimalloc::MiMalloc;
use std::io::{Read, Write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const FACTOR: usize = 1;

fn main() {
    // Get file name as first command line argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("Usage: {} <file> [<dest_size>]", args[0]);
        std::process::exit(1);
    }

    let dest_size = if args.len() == 3 {
        args[2].parse().unwrap()
    } else {
        4
    };
    // if dest_size is not power of 2, complain
    if dest_size & (dest_size - 1) != 0 {
        eprintln!("dest_size must be power of 2");
        std::process::exit(1);
    }

    let src_size: usize = dest_size * 2;

    // Ensure image is grayscale
    let img = image::open(&args[1]).unwrap().to_luma8();
    // Crash if image is not square
    assert_eq!(img.width(), img.height());

    // Erase contents of output directory
    std::fs::remove_dir_all("output").unwrap();
    std::fs::create_dir("output").unwrap();

    let compressed = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("output/compressed.leic")
        .unwrap();
    compress(img, &compressed, src_size, dest_size);
    drop(compressed);

    let mut compressed = std::fs::File::open("output/compressed.leic").unwrap();
    decompress(&mut compressed, src_size, dest_size);
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

    compressed
        .write_all(lzma::compress(data, 9 | EXTREME_PRESET).unwrap().as_slice())
        .unwrap();
    unsafe {
        std::alloc::dealloc(
            data.as_mut_ptr(),
            std::alloc::Layout::from_size_align(file_len, 8).unwrap(),
        );
    }
}

fn decompress(compressed: &mut std::fs::File, src_size: usize, dest_size: usize) {
    let file_len = compressed.metadata().unwrap().len() as usize;
    let mut data = Vec::new();
    compressed.read_to_end(&mut data).unwrap();
    let mut data = lzma::decompress(&data).unwrap();

    println!("lzma + fractal compressed file with size: {:>9}", file_len);
    println!(
        "       fractal compressed file with size: {:>9}",
        data.len()
    );

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

    for (i, iteration) in iterations.iter().enumerate() {
        let img = ImageBuffer::from_fn(
            iteration.nrows() as u32,
            iteration.ncols() as u32,
            |x, y| Luma([iteration[[y as usize, x as usize]] as u8]),
        );
        img.save(format!("output/output-{}.jpg", i)).unwrap();
    }
}
