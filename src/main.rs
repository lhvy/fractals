mod codec;

use image::{ImageBuffer, Luma};
use std::os::fd::AsRawFd;

type Float = f32;
type Index = u16;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

#[derive(Debug)]
struct Transformations<'a> {
    width: usize,
    height: usize,
    coordinate: &'a mut [Coordinate],
    is_flipped: &'a mut [u8],
    degrees: &'a mut [u8],
    adjustments: &'a mut [Adjustments],
    current: usize,
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
struct Coordinate {
    x: Index,
    y: Index,
}

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
