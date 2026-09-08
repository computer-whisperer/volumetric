//! Codegen helpers for wrapper bodies.
//!
//! Every helper appends to an [`InstrSeqBuilder`] so it works inside a
//! `block` closure as well as at the top of a function body. Positions and
//! bounds are `f64`s in the model's memory: a position is `n` values at a
//! pointer, bounds are `2 * n` interleaved `[min_0, max_0, ...]`.

use walrus::ir::{BinaryOp, LoadKind, MemArg, StoreKind};
use walrus::{InstrSeqBuilder, LocalId, MemoryId, ModuleLocals, ValType};

use crate::affine::Affine;

fn f64_at(index: usize) -> MemArg {
    MemArg {
        align: 3,
        offset: (index * 8) as u64,
    }
}

fn is_zero(v: f64) -> bool {
    v == 0.0
}

/// Load the first `n` f64s at `ptr` into fresh locals.
pub fn load_point(
    seq: &mut InstrSeqBuilder,
    locals: &mut ModuleLocals,
    memory: MemoryId,
    ptr: LocalId,
    n: usize,
) -> Vec<LocalId> {
    (0..n)
        .map(|i| {
            let local = locals.add(ValType::F64);
            seq.local_get(ptr)
                .load(memory, LoadKind::F64, f64_at(i))
                .local_set(local);
            local
        })
        .collect()
}

/// Store the point held in `src` mapped by `map` at `ptr`: for each axis
/// `i < src.len()`, `ptr[i] = sum_j map.linear[i][j] * src[j] + map.offset[i]`
/// over `j < src.len()`. Zero terms are skipped and unit factors elided, so
/// a translation costs one add per axis.
pub fn store_mapped_point(
    seq: &mut InstrSeqBuilder,
    memory: MemoryId,
    ptr: LocalId,
    src: &[LocalId],
    map: &Affine,
) {
    for i in 0..src.len() {
        seq.local_get(ptr);
        let mut terms = 0;
        for (j, &local) in src.iter().enumerate() {
            let factor = map.linear[i][j];
            if is_zero(factor) {
                continue;
            }
            seq.local_get(local);
            if factor != 1.0 {
                seq.f64_const(factor).binop(BinaryOp::F64Mul);
            }
            if terms > 0 {
                seq.binop(BinaryOp::F64Add);
            }
            terms += 1;
        }
        if terms == 0 {
            seq.f64_const(map.offset[i]);
        } else if !is_zero(map.offset[i]) {
            seq.f64_const(map.offset[i]).binop(BinaryOp::F64Add);
        }
        seq.store(memory, StoreKind::F64, f64_at(i));
    }
}

/// Rewrite the `n`-dimensional point at `ptr` in place by `map`.
pub fn map_point_in_place(
    seq: &mut InstrSeqBuilder,
    locals: &mut ModuleLocals,
    memory: MemoryId,
    ptr: LocalId,
    map: &Affine,
    n: usize,
) {
    let src = load_point(seq, locals, memory, ptr, n);
    store_mapped_point(seq, memory, ptr, &src, map);
}

/// An axis-aligned box held in locals: `min[i]`, `max[i]` per axis.
pub struct Bounds {
    pub min: Vec<LocalId>,
    pub max: Vec<LocalId>,
}

impl Bounds {
    pub fn new(locals: &mut ModuleLocals, n: usize) -> Self {
        Bounds {
            min: (0..n).map(|_| locals.add(ValType::F64)).collect(),
            max: (0..n).map(|_| locals.add(ValType::F64)).collect(),
        }
    }

    pub fn axes(&self) -> usize {
        self.min.len()
    }
}

/// Load the interleaved bounds at `ptr` into `dst` (its first `dst.axes()`
/// axes).
pub fn load_bounds(seq: &mut InstrSeqBuilder, memory: MemoryId, ptr: LocalId, dst: &Bounds) {
    for i in 0..dst.axes() {
        seq.local_get(ptr)
            .load(memory, LoadKind::F64, f64_at(2 * i))
            .local_set(dst.min[i]);
        seq.local_get(ptr)
            .load(memory, LoadKind::F64, f64_at(2 * i + 1))
            .local_set(dst.max[i]);
    }
}

/// Store `src` as interleaved bounds at `ptr`.
pub fn store_bounds(seq: &mut InstrSeqBuilder, memory: MemoryId, ptr: LocalId, src: &Bounds) {
    for i in 0..src.axes() {
        seq.local_get(ptr)
            .local_get(src.min[i])
            .store(memory, StoreKind::F64, f64_at(2 * i));
        seq.local_get(ptr)
            .local_get(src.max[i])
            .store(memory, StoreKind::F64, f64_at(2 * i + 1));
    }
}

/// Grow `acc` to enclose `next`.
pub fn fold_bounds(seq: &mut InstrSeqBuilder, acc: &Bounds, next: &Bounds) {
    for i in 0..acc.axes() {
        seq.local_get(acc.min[i])
            .local_get(next.min[i])
            .binop(BinaryOp::F64Min)
            .local_set(acc.min[i]);
        seq.local_get(acc.max[i])
            .local_get(next.max[i])
            .binop(BinaryOp::F64Max)
            .local_set(acc.max[i]);
    }
}

/// Maps a box through an affine map into the box enclosing its image,
/// with scratch locals allocated once so a body can map many boxes.
pub struct BoundsMapper {
    center: Vec<LocalId>,
    half: Vec<LocalId>,
    tmp_center: LocalId,
    tmp_half: LocalId,
}

impl BoundsMapper {
    pub fn new(locals: &mut ModuleLocals, n: usize) -> Self {
        BoundsMapper {
            center: (0..n).map(|_| locals.add(ValType::F64)).collect(),
            half: (0..n).map(|_| locals.add(ValType::F64)).collect(),
            tmp_center: locals.add(ValType::F64),
            tmp_half: locals.add(ValType::F64),
        }
    }

    /// Set `dst` to the axis-aligned box enclosing `map(src)` over the
    /// first `src.axes()` axes. `dst` may be `src`. A translation shifts
    /// min and max; a diagonal map scales them (swapping for negative
    /// factors, so the result stays exact); anything else goes through the
    /// centre/half-extent form `c' = M c + t`, `h' = |M| h`.
    pub fn map(&self, seq: &mut InstrSeqBuilder, src: &Bounds, map: &Affine, dst: &Bounds) {
        let n = src.axes();
        if map.is_translation() {
            for i in 0..n {
                for (from, to) in [(src.min[i], dst.min[i]), (src.max[i], dst.max[i])] {
                    seq.local_get(from);
                    if !is_zero(map.offset[i]) {
                        seq.f64_const(map.offset[i]).binop(BinaryOp::F64Add);
                    }
                    seq.local_set(to);
                }
            }
        } else if map.is_diagonal() {
            for i in 0..n {
                let factor = map.linear[i][i];
                for (from, to) in [(src.min[i], self.tmp_center), (src.max[i], self.tmp_half)] {
                    seq.local_get(from)
                        .f64_const(factor)
                        .binop(BinaryOp::F64Mul);
                    if !is_zero(map.offset[i]) {
                        seq.f64_const(map.offset[i]).binop(BinaryOp::F64Add);
                    }
                    seq.local_set(to);
                }
                seq.local_get(self.tmp_center)
                    .local_get(self.tmp_half)
                    .binop(BinaryOp::F64Min)
                    .local_set(dst.min[i]);
                seq.local_get(self.tmp_center)
                    .local_get(self.tmp_half)
                    .binop(BinaryOp::F64Max)
                    .local_set(dst.max[i]);
            }
        } else {
            for i in 0..n {
                seq.local_get(src.min[i])
                    .local_get(src.max[i])
                    .binop(BinaryOp::F64Add)
                    .f64_const(0.5)
                    .binop(BinaryOp::F64Mul)
                    .local_set(self.center[i]);
                seq.local_get(src.max[i])
                    .local_get(src.min[i])
                    .binop(BinaryOp::F64Sub)
                    .f64_const(0.5)
                    .binop(BinaryOp::F64Mul)
                    .local_set(self.half[i]);
            }
            for i in 0..n {
                // c'_i = sum_j m_ij c_j + t_i
                let mut terms = 0;
                for j in 0..n {
                    let factor = map.linear[i][j];
                    if is_zero(factor) {
                        continue;
                    }
                    seq.local_get(self.center[j]);
                    if factor != 1.0 {
                        seq.f64_const(factor).binop(BinaryOp::F64Mul);
                    }
                    if terms > 0 {
                        seq.binop(BinaryOp::F64Add);
                    }
                    terms += 1;
                }
                if terms == 0 {
                    seq.f64_const(map.offset[i]);
                } else if !is_zero(map.offset[i]) {
                    seq.f64_const(map.offset[i]).binop(BinaryOp::F64Add);
                }
                seq.local_set(self.tmp_center);
                // h'_i = sum_j |m_ij| h_j
                let mut terms = 0;
                for j in 0..n {
                    let factor = map.linear[i][j].abs();
                    if is_zero(factor) {
                        continue;
                    }
                    seq.local_get(self.half[j]);
                    if factor != 1.0 {
                        seq.f64_const(factor).binop(BinaryOp::F64Mul);
                    }
                    if terms > 0 {
                        seq.binop(BinaryOp::F64Add);
                    }
                    terms += 1;
                }
                if terms == 0 {
                    seq.f64_const(0.0);
                }
                seq.local_set(self.tmp_half);
                seq.local_get(self.tmp_center)
                    .local_get(self.tmp_half)
                    .binop(BinaryOp::F64Sub)
                    .local_set(dst.min[i]);
                seq.local_get(self.tmp_center)
                    .local_get(self.tmp_half)
                    .binop(BinaryOp::F64Add)
                    .local_set(dst.max[i]);
            }
        }
    }
}
