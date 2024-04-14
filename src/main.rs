mod codec;

use codec::{Adjustments, Coordinate, Transformations};
use image::{ImageBuffer, Luma};
use std::os::fd::AsRawFd;

const FACTOR: usize = 1;
const SRC_SIZE: usize = DEST_SIZE * 2;
const DEST_SIZE: usize = 8;
const STEP: usize = SRC_SIZE;

fn main() {
    // Get file name as first command line argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file>", args[0]);
        std::process::exit(1);
    }
    // Ensure image is grayscale
    let img = image::open(&args[1]).unwrap().to_luma8();
    // Crash if image is not square
    assert_eq!(img.width(), img.height());

    let compressed = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("output/compressed.leic")
        .unwrap();
    compress(img, &compressed);
    drop(compressed);

    let compressed = std::fs::File::open("output/compressed.leic").unwrap();
    decompress(compressed);
}

fn compress(img: ImageBuffer<Luma<u8>, Vec<u8>>, compressed: &std::fs::File) {
    let width = img.width() as usize / FACTOR / DEST_SIZE;
    let height = img.height() as usize / FACTOR / DEST_SIZE;
    let len = width * height;

    let header_size = std::mem::size_of::<usize>() * 2;
    let coordinate_size = std::mem::size_of::<Coordinate>() * len;
    let is_flipped_size = len.div_ceil(8);
    let degrees_size = len.div_ceil(4);
    let adjustments_size = std::mem::size_of::<Adjustments>() * len;
    let file_len =
        header_size + coordinate_size + is_flipped_size + degrees_size + adjustments_size;
    compressed.set_len(file_len as u64).unwrap();

    let data = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            file_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            compressed.as_raw_fd(),
            0,
        );
        // Error check
        if ptr == libc::MAP_FAILED {
            panic!("{}", *libc::__error());
        }

        std::slice::from_raw_parts_mut(ptr.cast::<u8>(), file_len)
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
        SRC_SIZE,
        DEST_SIZE,
        STEP,
        transformations,
    );
    unsafe {
        libc::msync(data.as_mut_ptr().cast(), file_len, libc::MS_SYNC);
        libc::munmap(data.as_mut_ptr().cast(), file_len);
    }
}

fn decompress(compressed: std::fs::File) {
    let file_len = compressed.metadata().unwrap().len() as usize;
    let data = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            file_len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            compressed.as_raw_fd(),
            0,
        );

        std::slice::from_raw_parts_mut(ptr.cast::<u8>(), file_len)
    };
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

    println!("Outputted compressed file with size: {}", file_len);

    let iterations = codec::decompress(t, SRC_SIZE, DEST_SIZE, STEP);

    for (i, iteration) in iterations.iter().enumerate() {
        let img = ImageBuffer::from_fn(
            iteration.nrows() as u32,
            iteration.ncols() as u32,
            |x, y| Luma([iteration[[y as usize, x as usize]] as u8]),
        );
        img.save(format!("output/output-{}.jpg", i)).unwrap();
    }
}
