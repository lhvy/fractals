use image::{ImageBuffer, Rgb};
use ndarray::{s, Array2, ArrayView2};

const MIN_DEPTH: usize = 1;
pub(crate) const DEFAULT_MAX_DEPTH: usize = 6;
pub(crate) const DEFAULT_DETAIL_THRESHOLD: f32 = 5.0;

pub(crate) struct Quadtree {
    width: usize,
    height: usize,
    root: Quadrant,
}

pub(crate) struct Storage {
    red: Array2<u8>,
    green: Array2<u8>,
    blue: Array2<u8>,
    max_depth: usize,
    detail_threshold: f32,
}

impl Storage {
    pub(crate) fn new(
        img: ImageBuffer<Rgb<u8>, Vec<u8>>,
        max_depth: usize,
        detail_threshold: f32,
    ) -> Self {
        let width = img.width() as usize;
        let height = img.height() as usize;
        let mut red = Array2::zeros((height, width));
        let mut green = Array2::zeros((height, width));
        let mut blue = Array2::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x as u32, y as u32);
                red[[y, x]] = pixel[0];
                green[[y, x]] = pixel[1];
                blue[[y, x]] = pixel[2];
            }
        }

        Self {
            red,
            green,
            blue,
            max_depth,
            detail_threshold,
        }
    }
}

impl Quadtree {
    pub(crate) fn new(storage: &Storage) -> Self {
        let width = storage.red.shape()[1];
        let height = storage.red.shape()[0];
        if (width / 2_usize.pow(storage.max_depth as u32) < 4)
            || (height / 2_usize.pow(storage.max_depth as u32) < 4)
        {
            panic!("Image is too small for the given depth. Aiming for at least 4x4 pixels per quadrant.");
        }
        let root = Quadrant::new(0, 0, width, height, 0, 0, storage);
        Self {
            width,
            height,
            root,
        }
    }

    pub(crate) fn leaves(&self) -> Vec<Quadrant> {
        let mut leaves = Vec::new();
        let mut stack = vec![&self.root];
        while let Some(quadrant) = stack.pop() {
            if let Some(children) = &quadrant.children {
                for child in children.iter() {
                    stack.push(child);
                }
            } else {
                leaves.push(quadrant.clone());
            }
        }

        leaves
    }

    pub(crate) fn create_image(&self, storage: &Storage) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let mut img = ImageBuffer::new(self.width as u32, self.height as u32);

        for leaf in self.leaves() {
            let range = s![leaf.y..leaf.y + leaf.height, leaf.x..leaf.x + leaf.width];
            let color = average(
                storage.red.slice(range),
                storage.green.slice(range),
                storage.blue.slice(range),
            );
            for y in 0..leaf.height {
                for x in 0..leaf.width {
                    img.put_pixel((leaf.x + x) as u32, (leaf.y + y) as u32, color);
                }
            }
        }

        img
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Quadrant {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
    pub(crate) index: usize,
    children: Option<Box<[Quadrant; 4]>>,
    detail: f32,
}

impl Quadrant {
    fn new(
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        depth: usize,
        index: usize,
        storage: &Storage,
    ) -> Self {
        let range = s![y..y + height, x..x + width];
        let detail = get_detail_intensity(
            &histogram(storage.red.slice(range)),
            &histogram(storage.green.slice(range)),
            &histogram(storage.blue.slice(range)),
        );
        let mut new = Self {
            x,
            y,
            width,
            height,
            depth,
            index,
            children: None,
            detail,
        };
        new.split(storage);

        new
    }

    fn max_index(&self) -> usize {
        if let Some(children) = &self.children {
            children
                .iter()
                .map(|child| child.max_index())
                .max()
                .unwrap()
        } else {
            self.index
        }
    }

    fn split(&mut self, storage: &Storage) {
        // println!("{}, {}, {}, {}", self.x, self.y, self.depth, self.index);
        if self.depth >= storage.max_depth
            || (self.detail < storage.detail_threshold && self.depth >= MIN_DEPTH)
        {
            return;
        }

        let width = self.width / 2;
        let height = self.height / 2;
        let depth = self.depth + 1;

        let top_left = Quadrant::new(
            self.x,
            self.y,
            width,
            height,
            depth,
            self.index + 1,
            storage,
        );
        let top_right = Quadrant::new(
            self.x + width,
            self.y,
            width,
            height,
            depth,
            top_left.max_index() + 1,
            storage,
        );
        let bottom_left = Quadrant::new(
            self.x,
            self.y + height,
            width,
            height,
            depth,
            top_right.max_index() + 1,
            storage,
        );
        let bottom_right = Quadrant::new(
            self.x + width,
            self.y + height,
            width,
            height,
            depth,
            bottom_left.max_index() + 1,
            storage,
        );
        let children = [top_left, top_right, bottom_left, bottom_right];
        self.children = Some(Box::new(children));

        // if let Some(children) = &mut self.children {
        //     for child in children.iter_mut() {
        //         child.split(storage);
        //     }
        // }
    }
}

fn average(red: ArrayView2<u8>, green: ArrayView2<u8>, blue: ArrayView2<u8>) -> Rgb<u8> {
    let total = red.len() as u32;
    let red = red.iter().map(|&value| value as u32).sum::<u32>() / total;
    let green = green.iter().map(|&value| value as u32).sum::<u32>() / total;
    let blue = blue.iter().map(|&value| value as u32).sum::<u32>() / total;
    Rgb([red as u8, green as u8, blue as u8])
}

fn histogram(data: ArrayView2<u8>) -> [u32; 256] {
    let mut histogram = [0; 256];
    for &value in data.iter() {
        histogram[value as usize] += 1;
    }
    histogram
}

fn weighted_error(histogram: &[u32; 256]) -> f32 {
    let mut error: f32 = 0.0;
    let total: u32 = histogram.iter().sum();

    if total > 0 {
        let value = histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| i as u32 * count)
            .sum::<u32>()
            / total;
        error = (histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| count as i64 * (value as i64 - i as i64).pow(2))
            .sum::<i64>()) as f32
            / total as f32;
    }

    error.sqrt()
}

fn get_detail_intensity(
    red_histogram: &[u32; 256],
    green_histogram: &[u32; 256],
    blue_histogram: &[u32; 256],
) -> f32 {
    let red_error = weighted_error(red_histogram);
    let green_error = weighted_error(green_histogram);
    let blue_error = weighted_error(blue_histogram);

    red_error * 0.2989 + green_error * 0.5870 + blue_error * 0.1140
}
