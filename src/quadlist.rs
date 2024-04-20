use crate::codec::Index;
use std::cmp::Ordering;

struct QuadrantIterator {
    image_diameter: Index,
    steps: Vec<QuadrantStep>,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum QuadrantStep {
    TopLeft = 0b00,
    TopRight = 0b01,
    BottomLeft = 0b10,
    BottomRight = 0b11,
}

impl QuadrantIterator {
    fn new(image_diameter: Index) -> Self {
        Self {
            image_diameter,
            steps: Vec::new(),
        }
    }

    fn step(&mut self, depth: usize) -> (Index, Index) {
        match self.steps.len().cmp(&depth) {
            Ordering::Greater => self.up(),
            Ordering::Equal => self.next(),
            Ordering::Less => self.deeper(depth - self.steps.len()),
        }

        dbg!(depth, self.current_position(), &self.steps);
        self.current_position()
    }

    fn up(&mut self) {
        assert_eq!(*self.steps.last().unwrap(), QuadrantStep::BottomRight);

        while let Some(QuadrantStep::BottomRight) = self.steps.last().copied() {
            self.steps.pop();
        }

        if let Some(step) = self.steps.last().copied() {
            *self.steps.last_mut().unwrap() = step.next()
        }
    }

    fn deeper(&mut self, depth: usize) {
        self.next();

        for _ in 0..depth {
            self.steps.push(QuadrantStep::TopLeft);
        }
    }

    fn next(&mut self) {
        let mut i = 0;
        while let Some(QuadrantStep::BottomRight) = self.steps.last().copied() {
            self.steps.pop();
            i += 1;
        }

        if let Some(step) = self.steps.last().copied() {
            *self.steps.last_mut().unwrap() = step.next()
        }

        for _ in 0..i {
            self.steps.push(QuadrantStep::TopLeft);
        }
    }

    fn current_position(&self) -> (Index, Index) {
        let mut x = 0;
        let mut y = 0;

        for (i, step) in self.steps.iter().enumerate() {
            let current_quadrant_diameter = self.image_diameter >> (i + 1);

            let step_u8 = *step as u8;
            let going_right = (step_u8 & 0b01 != 0) as Index;
            let going_down = (step_u8 & 0b10 != 0) as Index;
            x += current_quadrant_diameter * going_right;
            y += current_quadrant_diameter * going_down;
        }

        (x, y)
    }
}

impl QuadrantStep {
    fn next(self) -> QuadrantStep {
        let self_u8 = self as u8;
        let next_u8 = (self_u8 + 1) & 0b11;
        unsafe { std::mem::transmute::<u8, QuadrantStep>(next_u8) }
    }
}

#[cfg(test)]
#[test]
fn test() -> std::io::Result<()> {
    use std::fs;

    for entry in fs::read_dir("quadrant_iterator_test_data")? {
        let entry = entry?;
        let path = entry.path().canonicalize()?;
        let contents = fs::read_to_string(&path)?;
        let contents = contents.trim();

        let (_, input) = contents.split_once("\n\n").unwrap();

        let (size, depths) = input.split_once('\n').unwrap();
        let (width, height) = size.split_once('x').unwrap();
        assert_eq!(width, height);
        let diameter = width.parse().unwrap();

        let depths = depths.split(',').map(|depth| depth.parse().unwrap());

        let mut iterator = QuadrantIterator::new(diameter);
        let mut actual_output = String::new();
        for depth in depths {
            let (x, y) = iterator.step(depth);
            actual_output.push_str(&format!("({x}, {y})\n"));
        }

        let actual_contents = format!("{actual_output}\n{input}\n");

        let expect = expect_test::expect_file![path];
        expect.assert_eq(&actual_contents);
    }

    Ok(())
}
