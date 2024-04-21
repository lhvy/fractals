mod codec;
mod quadlist;
mod quadtree;

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
    // Get file name as first command line argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        eprintln!(
            "Usage: {} <file> [<max_depth>] [<detail_threshold>]",
            args[0]
        );
        std::process::exit(1);
    }

    let max_depth = args
        .get(2)
        .map_or(quadtree::DEFAULT_MAX_DEPTH, |s| s.parse().unwrap());
    let detail_threshold = args
        .get(3)
        .map_or(quadtree::DEFAULT_DETAIL_THRESHOLD, |s| s.parse().unwrap());

    // Open image as RGB
    let img = image::open(&args[1]).unwrap().to_rgb8();
    let storage = quadtree::Storage::new(img, max_depth, detail_threshold);
    let quadtree = quadtree::Quadtree::new(&storage);
    let mut leaves = quadtree.leaves();

    // Sort leaves by leave.index
    leaves.sort_by_key(|leave| leave.index);
    // Indexes are increasing, but make them consecutive starting from 0
    for (i, leave) in leaves.iter_mut().enumerate() {
        leave.index = i;
    }

    // Ensure image is grayscale
    let img = image::open(&args[1]).unwrap().to_luma8();
    // Crash if image is not square
    assert_eq!(img.width(), img.height());

    // Erase contents of output directory
    std::fs::remove_dir_all("output").unwrap();
    std::fs::create_dir("output").unwrap();

    quadtree
        .create_image(&storage)
        .save("output/quadtree.jpg")
        .unwrap();

    let compressed = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("output/compressed.leic")
        .unwrap();
    compress(img, &leaves, &compressed);
    drop(compressed);

    let mut compressed = std::fs::File::open("output/compressed.leic").unwrap();
    decompress(&mut compressed);
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

    codec::compress(codec::reduce(&img, FACTOR), leaves, transformations);

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

fn decompress(compressed: &mut std::fs::File) {
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

    for (i, iteration) in iterations.iter().enumerate() {
        let img = ImageBuffer::from_fn(
            iteration.nrows() as u32,
            iteration.ncols() as u32,
            |x, y| Luma([iteration[[y as usize, x as usize]] as u8]),
        );
        img.save(format!("output/output-{}.jpg", i)).unwrap();
    }
}
