use image::{ImageBuffer, Rgb};
use ndarray::{s, Array2, ArrayView2};

const MIN_DEPTH: usize = 1;
const MAX_DEPTH: usize = 6;
const DETAIL_THRESHOLD: f32 = 5.0;

pub(crate) struct Quadtree {
    width: usize,
    height: usize,
    depth: usize,
    root: Quadrant,
}

pub(crate) struct Storage {
    red: Array2<u8>,
    green: Array2<u8>,
    blue: Array2<u8>,
}

impl Storage {
    pub(crate) fn new(img: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
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

        Self { red, green, blue }
    }
}

impl Quadtree {
    pub(crate) fn new(storage: &Storage) -> Self {
        let width = storage.red.shape()[1];
        let height = storage.red.shape()[0];
        let root = Quadrant::new(0, 0, width, height, 0, 0, storage);
        Self {
            width,
            height,
            depth: 0,
            root,
        }
    }

    pub(crate) fn build(&mut self, storage: &Storage) {
        self.root.split(storage);
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

#[derive(Clone)]
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
        Self {
            x,
            y,
            width,
            height,
            depth,
            index,
            children: None,
            detail,
        }
    }

    fn split(&mut self, storage: &Storage) {
        if self.depth >= MAX_DEPTH || (self.detail < DETAIL_THRESHOLD && self.depth >= MIN_DEPTH) {
            return;
        }

        let width = self.width / 2;
        let height = self.height / 2;
        let depth = self.depth + 1;
        let children = [
            Quadrant::new(
                self.x,
                self.y,
                width,
                height,
                depth,
                self.index + 1,
                storage,
            ),
            Quadrant::new(
                self.x + width,
                self.y,
                width,
                height,
                depth,
                self.index + 2,
                storage,
            ),
            Quadrant::new(
                self.x,
                self.y + height,
                width,
                height,
                depth,
                self.index + 3,
                storage,
            ),
            Quadrant::new(
                self.x + width,
                self.y + height,
                width,
                height,
                depth,
                self.index + 4,
                storage,
            ),
        ];
        self.children = Some(Box::new(children));

        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                child.split(storage);
            }
        }
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
