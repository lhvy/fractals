use super::{
    reduce_block, transform, Adjustments, Coordinate, Float, Index, Rotation, Transformation,
    Transformations,
};
use ndarray::{s, Array2};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct Cache {
    pub(crate) map: HashMap<u8, Slot>,
}

pub(crate) struct Slot {
    pub(crate) transformations: Transformations<'static>,
    pub(crate) src_blocks: Box<[Array2<Float>]>,
}

impl Cache {
    pub(crate) fn get(&mut self, m: &Array2<Float>, depth: u8) -> &Slot {
        let entry = self.map.entry(depth);
        match entry {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                let (t, src_blocks) = gen_all_transformations(m, depth);

                v.insert(Slot {
                    transformations: t,
                    src_blocks: src_blocks.into_boxed_slice(),
                })
            }
        }
    }
}

fn gen_all_transformations(
    m: &Array2<Float>,
    depth: u8,
) -> (Transformations<'static>, Vec<Array2<Float>>) {
    let mut blocks = Vec::new();

    // -1 because source blocks are double dest blocks
    let n_rows = 2_usize.pow(depth as u32 - 1);
    let n_cols = 2_usize.pow(depth as u32 - 1);
    let src_height = m.nrows() / n_rows;
    let src_width = m.ncols() / n_cols;
    let mut transformations: Transformations<'static> = unsafe {
        let n = n_cols * n_rows * 2 * 4;
        Transformations {
            header: super::Header::default(),
            src_coordinate: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(std::alloc::Layout::array::<Coordinate>(n).unwrap())
                    .cast(),
                n,
            ),
            scale: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(std::alloc::Layout::array::<u8>(n).unwrap()).cast(),
                n,
            ),
            is_flipped: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(std::alloc::Layout::array::<u8>((n).div_ceil(8)).unwrap()),
                (n).div_ceil(8),
            ),
            degrees: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(std::alloc::Layout::array::<u8>((n).div_ceil(4)).unwrap()),
                (n).div_ceil(4),
            ),
            adjustments: std::slice::from_raw_parts_mut(
                std::alloc::alloc_zeroed(std::alloc::Layout::array::<Adjustments>(n).unwrap())
                    .cast(),
                n,
            ),
            current: 0,
        }
    };

    for y in (0..m.nrows()).step_by(src_height) {
        for x in (0..m.ncols()).step_by(src_width) {
            let src_block = reduce_block(
                m.clone()
                    .slice_move(s![y..y + src_height, x..x + src_width]),
                2,
            );
            for is_flipped in [false, true] {
                for degrees in [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270] {
                    let t = Transformation {
                        src_x: x as Index,
                        src_y: y as Index,
                        scale: depth,
                        is_flipped,
                        degrees,
                        adjustments: Adjustments::default(),
                    };
                    blocks.push(transform(src_block.clone(), &t));
                    transformations.push(t);
                }
            }
        }
    }

    (transformations, blocks)
}
