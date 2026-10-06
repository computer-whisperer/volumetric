//! Direct casting: finding a model's surface by sampling it along view
//! rays, without building a mesh. The design and the reasons for it are in
//! `DIRECT_CASTING_PLAN.md`.
//!
//! A model answers one question, "is this point inside?", so a ray finds
//! the surface by stepping until a sample is inside and bisecting back.
//! Everything found is kept in a [`DirectCast`], an octree over the
//! model's bounds holding two things in the model's own frame:
//!
//! - **surface points** ([`Surfel`]), each at the level of the tree whose
//!   cell size matches the pitch it was found at, so the tree is a
//!   level-of-detail hierarchy of the surface;
//! - **the search record**: which cells hold surface, and how finely each
//!   empty cell has been searched, so later rays (from any view) skip
//!   space already searched at the pitch they need.
//!
//! One [`DirectCast::pass`] casts a lattice of rays in one of three
//! [`PassMode`]s; [`DirectCast::cast`] runs the passes that take a view
//! from nothing to a finished image.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use glam::{DVec3, Vec3};

use crate::wasm::ParallelModelSampler;

/// A node's nominal pitch is its side over this: the node is as wide as
/// this many of the samples that found its surface points.
const CELLS_PER_NODE: f64 = 4.0;
/// Bisection steps per surface point. Each halves the bracket, which
/// starts one pitch wide.
const BISECTIONS: usize = 10;
/// The tree starts split to this level, so the first pass already records
/// what it searched in pieces smaller than the whole model.
const FIRST_LEVEL: u32 = 3;
const MAX_LEVEL: u32 = 24;
/// An empty node counts as searched by a pass when at least this fraction
/// of the rays its outline should catch went all the way through it.
const COVERAGE: f64 = 0.7;
/// A node searched at up to this many times the pitch a ray wants is taken
/// as searched. Without the allowance a view a little closer than the one
/// before would search everything again for a few percent more certainty.
const SEARCH_SLACK: f64 = 1.5;
/// A hit is on something thin, or at an edge of what is known, when a ray
/// beside it missed or hit more than this many pitches nearer or farther.
const ISOLATED_DEPTH: f64 = 4.0;
/// Two pitches within this factor count as the same. Passes a factor of
/// two apart in spacing can land surfels in the same level of the tree,
/// and the coarser must not stand in for the finer.
const SAME_PITCH: f64 = 1.2;
/// A surfel's disc radius over the pitch it was found at: enough for
/// discs a pitch apart on a square lattice to leave no gaps.
const RADIUS_PER_PITCH: f64 = 0.75;

/// How a view turns pixels into rays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CastProjection {
    Perspective { tan_half_fov_y: f64 },
    Orthographic { half_height: f64 },
}

/// A camera and an image size. `right`, `up` and `forward` are
/// orthonormal; `eye` is the centre of projection, or for an orthographic
/// view the centre of the image plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastView {
    pub eye: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    pub forward: DVec3,
    pub projection: CastProjection,
    pub width: u32,
    pub height: u32,
}

impl CastView {
    pub fn look_at(
        eye: DVec3,
        target: DVec3,
        up_hint: DVec3,
        projection: CastProjection,
        width: u32,
        height: u32,
    ) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(up_hint).normalize();
        Self {
            eye,
            right,
            up: right.cross(forward),
            forward,
            projection,
            width,
            height,
        }
    }

    /// The ray through image position (`x`, `y`) in pixels, origin at the
    /// top left.
    pub fn ray(&self, x: f64, y: f64) -> Ray {
        let (w, h) = (self.width as f64, self.height as f64);
        let u = (2.0 * x / w - 1.0) * (w / h);
        let v = 1.0 - 2.0 * y / h;
        match self.projection {
            CastProjection::Perspective { tan_half_fov_y } => {
                let through = self.forward
                    + self.right * (u * tan_half_fov_y)
                    + self.up * (v * tan_half_fov_y);
                let length = through.length();
                Ray {
                    origin: self.eye,
                    dir: through / length,
                    footprint_base: 0.0,
                    footprint_slope: 2.0 * tan_half_fov_y / (h * length),
                }
            }
            CastProjection::Orthographic { half_height } => Ray {
                origin: self.eye + self.right * (u * half_height) + self.up * (v * half_height),
                dir: self.forward,
                footprint_base: 2.0 * half_height / h,
                footprint_slope: 0.0,
            },
        }
    }

    /// The world size of one pixel at `point`.
    pub fn footprint_at(&self, point: DVec3) -> f64 {
        match self.projection {
            CastProjection::Perspective { tan_half_fov_y } => {
                (point - self.eye).dot(self.forward).max(0.0) * 2.0 * tan_half_fov_y
                    / self.height as f64
            }
            CastProjection::Orthographic { half_height } => 2.0 * half_height / self.height as f64,
        }
    }

    /// Where `point` falls in the image, in pixels, if it is in front of
    /// the view and inside the frame.
    pub fn pixel_of(&self, point: DVec3) -> Option<(f64, f64)> {
        let offset = point - self.eye;
        let depth = offset.dot(self.forward);
        let scale = match self.projection {
            CastProjection::Perspective { tan_half_fov_y } => {
                if depth <= 0.0 {
                    return None;
                }
                depth * tan_half_fov_y
            }
            CastProjection::Orthographic { half_height } => half_height,
        };
        let (w, h) = (self.width as f64, self.height as f64);
        let x = (offset.dot(self.right) / scale * (h / w) + 1.0) * 0.5 * w;
        let y = (1.0 - offset.dot(self.up) / scale) * 0.5 * h;
        (x >= 0.0 && y >= 0.0 && x < w && y < h).then_some((x, y))
    }

    /// The unit direction rays travel at `point`.
    fn direction_at(&self, point: DVec3) -> DVec3 {
        match self.projection {
            CastProjection::Perspective { .. } => (point - self.eye).normalize_or(self.forward),
            CastProjection::Orthographic { .. } => self.forward,
        }
    }
}

/// A view ray with the pixel footprint along it.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    origin: DVec3,
    dir: DVec3,
    footprint_base: f64,
    footprint_slope: f64,
}

impl Ray {
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }

    /// The world size of one pixel at parameter `t`.
    fn footprint(&self, t: f64) -> f64 {
        self.footprint_base + self.footprint_slope * t
    }
}

/// A point on the model's surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surfel {
    pub position: [f32; 3],
    /// Unit, pointing out of the model.
    pub normal: [f32; 3],
    /// Radius of the disc that stands for the surface around the point.
    pub radius: f32,
}

/// What a pass steps through. Every mode steps nodes known to hold
/// surface; they differ in which empty space they also search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassMode {
    /// Also searches every node not yet searched at the pass's pitch.
    /// This is the pass that finds things, and the expensive one.
    Discover,
    /// Also searches the unsearched nodes of its own pitch that share a
    /// parent with known surface, where the surface most likely continues. Cheap: it
    /// redraws known surface at a finer lattice.
    Refine,
    /// Searches only those neighbouring nodes, in front of what the image
    /// already shows, and marks the neighbours of whatever it finds in
    /// turn: run until it finds nothing, it follows a thin thing along
    /// its length from wherever a search happened on it.
    Chase,
}

/// Where one ray met the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the ray.
    pub t: f64,
    pub position: DVec3,
    /// Unit, pointing out of the model.
    pub normal: Vec3,
}

/// The rays of one pass: a lattice of `width × height` rays `spacing`
/// pixels apart, the first half a spacing in from the top left.
#[derive(Clone, Debug, PartialEq)]
pub struct CastImage {
    pub width: u32,
    pub height: u32,
    pub spacing: f64,
    pub hits: Vec<Option<RayHit>>,
}

impl CastImage {
    fn lattice(view: &CastView, spacing: f64) -> (u32, u32) {
        (
            (view.width as f64 / spacing).ceil() as u32,
            (view.height as f64 / spacing).ceil() as u32,
        )
    }

    pub fn hit(&self, x: u32, y: u32) -> Option<&RayHit> {
        self.hits[(y * self.width + x) as usize].as_ref()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PassStats {
    pub rays: u64,
    pub samples: u64,
    pub hits: u64,
    /// Surface points the tree did not have before.
    pub fresh: u64,
    /// Of those, the ones that had the nodes around them marked for a
    /// closer look.
    pub followed: u64,
    pub seconds: f64,
}

impl std::ops::AddAssign for PassStats {
    fn add_assign(&mut self, other: Self) {
        self.rays += other.rays;
        self.samples += other.samples;
        self.hits += other.hits;
        self.fresh += other.fresh;
        self.followed += other.followed;
        self.seconds += other.seconds;
    }
}

/// The passes [`DirectCast::cast`] runs, all in pixels of the view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastOptions {
    /// Ray spacing of the finished image. Below 1 supersamples.
    pub spacing: f64,
    /// Ray spacing of the first pass; each later level halves it.
    pub coarsest: f64,
    /// The finest pitch empty space is searched at. At `spacing`, nothing
    /// a pixel wide is missed; larger is faster and can miss things
    /// thinner than this many pixels that no coarser pass happened on.
    pub search: f64,
}

impl Default for CastOptions {
    fn default() -> Self {
        Self {
            spacing: 1.0,
            coarsest: 8.0,
            search: 1.0,
        }
    }
}

struct Node {
    /// Index of the first of eight children, 0 for a leaf.
    first_child: u32,
    /// The finest pitch this node's volume has been searched at and found
    /// empty. Holds for every node beneath it too.
    searched: f32,
    /// A surface point has been found in this node or beneath it.
    occupied: bool,
    /// Next to a node where something thin was found and not yet
    /// searched at its own pitch: the next pass to reach it steps through
    /// it, whatever its mode.
    suspect: bool,
    /// A ray of the pass in progress stepped into this suspect node.
    reached: AtomicBool,
    /// A sample of the pass in progress taken in this node was inside the
    /// model.
    struck: AtomicBool,
    /// Rays of the pass in progress that stepped all the way through
    /// without a hit.
    through: AtomicU32,
    /// Surface points found at this node's pitch.
    surfels: Vec<Surfel>,
}

impl Node {
    fn new(searched: f32) -> Self {
        Self {
            first_child: 0,
            searched,
            occupied: false,
            suspect: false,
            reached: AtomicBool::new(false),
            struck: AtomicBool::new(false),
            through: AtomicU32::new(0),
            surfels: Vec::new(),
        }
    }
}

/// A node found by descending the tree, with the cube it covers.
#[derive(Clone, Copy)]
struct Located {
    index: u32,
    center: DVec3,
    half: f64,
    /// Finest pitch the node is known empty at: the least `searched` on
    /// the way down.
    searched: f32,
    /// The node's parent holds surface somewhere.
    beside_surface: bool,
}

impl Located {
    /// The eight children, for walks that do not track `searched`.
    fn children(self, first_child: u32) -> impl Iterator<Item = Located> {
        let half = self.half * 0.5;
        (0..8u32).map(move |octant| {
            let sign = |bit: u32| if octant & bit != 0 { half } else { -half };
            Located {
                index: first_child + octant,
                center: self.center + DVec3::new(sign(1), sign(2), sign(4)),
                half,
                searched: self.searched,
                beside_surface: false,
            }
        })
    }

    /// Where `ray` leaves the node.
    fn exit(&self, ray: &Ray) -> f64 {
        let mut exit = f64::INFINITY;
        for axis in 0..3 {
            let dir = ray.dir[axis];
            if dir != 0.0 {
                let face = self.center[axis] + self.half.copysign(dir);
                exit = exit.min((face - ray.origin[axis]) / dir);
            }
        }
        exit
    }
}

/// A surface point a pass found that the tree did not have.
struct Fresh {
    position: DVec3,
    normal: Vec3,
    pitch: f64,
    /// Distance along its ray.
    t: f64,
}

/// Everything known about one model's surface and the space around it.
/// See the module documentation.
pub struct DirectCast {
    bounds_min: DVec3,
    bounds_max: DVec3,
    center: DVec3,
    half: f64,
    nodes: Vec<Node>,
    passes: u32,
    pub total: PassStats,
    /// Every pass run, in order, with what it cost.
    pub history: Vec<(PassMode, f64, PassStats)>,
}

impl DirectCast {
    /// An empty record for a model with the given bounds.
    pub fn new(bounds_min: DVec3, bounds_max: DVec3) -> anyhow::Result<Self> {
        anyhow::ensure!(
            bounds_min.is_finite() && bounds_max.is_finite() && bounds_max.cmpgt(bounds_min).all(),
            "model reported invalid bounds"
        );
        let mut cast = Self {
            bounds_min,
            bounds_max,
            center: (bounds_min + bounds_max) * 0.5,
            half: (bounds_max - bounds_min).max_element() * 0.5 * 1.001,
            nodes: vec![Node::new(f32::INFINITY)],
            passes: 0,
            total: PassStats::default(),
            history: Vec::new(),
        };
        cast.split_to(0, FIRST_LEVEL);
        Ok(cast)
    }

    /// An empty record for `sampler`'s model.
    pub fn for_model(sampler: &impl ParallelModelSampler) -> anyhow::Result<Self> {
        let bounds = sampler.get_bounds()?;
        Self::new(bounds.min.into(), bounds.max.into())
    }

    fn split_to(&mut self, index: u32, levels: u32) {
        if levels == 0 {
            return;
        }
        let first = self.split(index);
        for child in first..first + 8 {
            self.split_to(child, levels - 1);
        }
    }

    /// Gives a leaf its eight children and returns the first. What was
    /// known of the leaf's whole volume is known of each child.
    fn split(&mut self, index: u32) -> u32 {
        let first = self.nodes.len() as u32;
        let searched = self.nodes[index as usize].searched;
        self.nodes.extend((0..8).map(|_| Node::new(searched)));
        self.nodes[index as usize].first_child = first;
        first
    }

    fn pitch_of(half: f64) -> f64 {
        2.0 * half / CELLS_PER_NODE
    }

    /// The level whose nodes have a pitch no coarser than `pitch`.
    fn level_for(&self, pitch: f64) -> u32 {
        let level = (2.0 * self.half / (CELLS_PER_NODE * pitch)).log2().ceil();
        (level.max(FIRST_LEVEL as f64) as u32).min(MAX_LEVEL)
    }

    fn child_of(node: &Located, first_child: u32, point: DVec3) -> (u32, DVec3, f64) {
        let above = point.cmpge(node.center);
        let half = node.half * 0.5;
        let offset = DVec3::select(above, DVec3::splat(half), DVec3::splat(-half));
        (first_child + above.bitmask(), node.center + offset, half)
    }

    fn root(&self) -> Located {
        Located {
            index: 0,
            center: self.center,
            half: self.half,
            searched: self.nodes[0].searched,
            beside_surface: false,
        }
    }

    /// The node holding `point` that a ray wanting `pitch` should treat as
    /// one piece: the first on the way down that is a leaf or fine enough.
    fn locate(&self, point: DVec3, pitch: f64) -> Located {
        let mut at = self.root();
        loop {
            let node = &self.nodes[at.index as usize];
            if node.first_child == 0 || Self::pitch_of(at.half) <= pitch {
                return at;
            }
            let (index, center, half) = Self::child_of(&at, node.first_child, point);
            at = Located {
                index,
                center,
                half,
                searched: at.searched.min(self.nodes[index as usize].searched),
                beside_surface: self.nodes[at.index as usize].occupied,
            };
        }
    }

    /// The node at exactly `level` holding `point`, if the tree goes that
    /// deep there.
    fn node_at(&self, point: DVec3, level: u32) -> Option<Located> {
        let mut at = self.root();
        for _ in 0..level {
            let node = &self.nodes[at.index as usize];
            if node.first_child == 0 {
                return None;
            }
            let (index, center, half) = Self::child_of(&at, node.first_child, point);
            at = Located {
                index,
                center,
                half,
                searched: at.searched.min(self.nodes[index as usize].searched),
                beside_surface: self.nodes[at.index as usize].occupied,
            };
        }
        Some(at)
    }

    /// The node at exactly `level` holding `point`, splitting leaves on
    /// the way down as needed.
    fn ensure(&mut self, point: DVec3, level: u32) -> Located {
        let mut at = self.root();
        for _ in 0..level {
            let mut first_child = self.nodes[at.index as usize].first_child;
            if first_child == 0 {
                first_child = self.split(at.index);
            }
            let (index, center, half) = Self::child_of(&at, first_child, point);
            at = Located {
                index,
                center,
                half,
                searched: at.searched.min(self.nodes[index as usize].searched),
                beside_surface: self.nodes[at.index as usize].occupied,
            };
        }
        at
    }

    /// A surfel found at about `pitch` or finer, within half of it of
    /// `point`, that faces a ray travelling along `dir`.
    fn surfel_near(&self, point: DVec3, pitch: f64, dir: DVec3) -> Option<&Surfel> {
        let node = self.node_at(point, self.level_for(pitch))?;
        let reach = 0.5 * pitch;
        self.nodes[node.index as usize].surfels.iter().find(|s| {
            let position = Vec3::from(s.position).as_dvec3();
            (s.radius as f64) < RADIUS_PER_PITCH * pitch * SAME_PITCH
                && position.distance_squared(point) < reach * reach
                && Vec3::from(s.normal).as_dvec3().dot(dir) < 0.0
        })
    }

    /// Whether any surfel, at any level, lies within `reach` of `point`.
    pub fn has_surfel_near(&self, point: DVec3, reach: f64) -> bool {
        let mut stack = vec![self.root()];
        while let Some(at) = stack.pop() {
            if (point - at.center).abs().max_element() > at.half + reach {
                continue;
            }
            let node = &self.nodes[at.index as usize];
            if node
                .surfels
                .iter()
                .any(|s| Vec3::from(s.position).as_dvec3().distance_squared(point) < reach * reach)
            {
                return true;
            }
            if node.first_child != 0 {
                stack.extend(at.children(node.first_child));
            }
        }
        false
    }

    /// The surfels to draw. With a view, each part of the surface comes
    /// from the level whose pitch suits `spacing` pixels there, or from
    /// finer levels where that one has nothing; without, from the finest
    /// level everywhere. A coarser surfel stands in only where nothing
    /// finer has been found beneath it.
    pub fn surfels(&self, view: Option<(&CastView, f64)>) -> Vec<Surfel> {
        let mut out = Vec::new();
        let mut stack = vec![self.root()];
        while let Some(at) = stack.pop() {
            let node = &self.nodes[at.index as usize];
            if !node.occupied {
                continue;
            }
            let wanted = view.map_or(0.0, |(view, spacing)| {
                spacing * view.footprint_at(at.center)
            });
            let fine_enough = Self::pitch_of(at.half) <= wanted && !node.surfels.is_empty();
            if fine_enough || node.first_child == 0 {
                out.extend(node.surfels.iter().copied());
                continue;
            }
            out.extend(
                node.surfels
                    .iter()
                    .filter(|s| !self.refined_below(&at, Vec3::from(s.position).as_dvec3()))
                    .copied(),
            );
            stack.extend(at.children(node.first_child));
        }
        out
    }

    /// Whether a node beneath `at` holding `point` has surfels of its own.
    fn refined_below(&self, at: &Located, point: DVec3) -> bool {
        let mut walk = *at;
        loop {
            let node = &self.nodes[walk.index as usize];
            if node.first_child == 0 {
                return false;
            }
            let (index, center, half) = Self::child_of(&walk, node.first_child, point);
            walk = Located {
                index,
                center,
                half,
                ..walk
            };
            if !self.nodes[walk.index as usize].surfels.is_empty() {
                return true;
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Where `ray` crosses the model's bounds.
    fn span(&self, ray: &Ray) -> Option<(f64, f64)> {
        let (mut near, mut far) = (0.0f64, f64::INFINITY);
        for axis in 0..3 {
            let (low, high) = (self.bounds_min[axis], self.bounds_max[axis]);
            let (origin, dir) = (ray.origin[axis], ray.dir[axis]);
            if dir == 0.0 {
                if origin < low || origin > high {
                    return None;
                }
                continue;
            }
            let (a, b) = ((low - origin) / dir, (high - origin) / dir);
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
        (near < far).then_some((near, far))
    }

    /// Follows one ray front to back as far as `limit` and returns its
    /// first hit, with any surface point the tree lacked.
    #[allow(clippy::too_many_arguments)]
    fn trace(
        &self,
        sampler: &(impl ParallelModelSampler + ?Sized),
        view: &CastView,
        ray: &Ray,
        spacing: f64,
        mode: PassMode,
        jitter: f64,
        limit: f64,
        samples: &mut u64,
    ) -> Option<(RayHit, Option<Fresh>)> {
        let (enter, leave) = self.span(ray)?;
        let leave = leave.min(limit);
        let nudge = self.half * 1e-9;
        let mut inside = |t: f64| {
            *samples += 1;
            let p = ray.at(t);
            volumetric_abi::is_occupied(sampler.sample(p.x, p.y, p.z))
        };

        // A model that reaches its bounds has surface there: beyond them
        // is outside by definition. Something thinner than a step would
        // otherwise be stepped over at the very start.
        // (A chase adds to an image that already has these hits.)
        if mode != PassMode::Chase && enter > 0.0 && enter < leave && inside(enter + nudge) {
            let position = ray.at(enter);
            let off = ((position - self.bounds_min).abs()).min((position - self.bounds_max).abs());
            let axis = off.min_position();
            let mut normal = Vec3::ZERO;
            normal[axis] = -ray.dir[axis].signum() as f32;
            let hit = RayHit {
                t: enter,
                position,
                normal,
            };
            let pitch = spacing * ray.footprint(enter);
            let fresh = self
                .surfel_near(position, pitch, ray.dir)
                .is_none()
                .then_some(Fresh {
                    position,
                    normal,
                    pitch,
                    t: enter,
                });
            return Some((hit, fresh));
        }

        let mut t = enter;
        // Where the run of steps in progress samples next, if the node
        // just left was stepped.
        let mut next: Option<f64> = None;
        // The last place known to be outside the model. The ray enters the
        // bounds from outside them.
        let mut outside = enter;
        let mut was_stepped = false;
        while t < leave {
            let pitch = spacing * ray.footprint(t);
            let at = self.locate(ray.at(t + nudge), pitch);
            let node = &self.nodes[at.index as usize];
            let exit = at.exit(ray);
            let stop = exit.min(leave);

            let stepped = match mode {
                PassMode::Discover => {
                    node.occupied || node.suspect || at.searched as f64 > pitch * SEARCH_SLACK
                }
                PassMode::Refine => {
                    node.occupied
                        || node.suspect
                        || (at.beside_surface
                            && Self::pitch_of(at.half) <= pitch
                            && at.searched as f64 > pitch * SEARCH_SLACK)
                }
                PassMode::Chase => node.suspect,
            };
            // Where stepping starts or stops, the boundary itself is
            // sampled. Skipping is decided node by node, and without this
            // a ray crosses unseen through anything that lies partly in a
            // stepped node, too little of it to be met by a step, and
            // partly in a skipped one.
            let mut found = None;
            if stepped != was_stepped && t > enter {
                let boundary = t + nudge;
                if inside(boundary) {
                    found = Some(boundary);
                } else {
                    outside = boundary;
                }
            }
            was_stepped = stepped;
            if !stepped && found.is_none() {
                next = None;
                t = exit.max(t + nudge);
                continue;
            }

            if node.suspect {
                node.reached.store(true, Ordering::Relaxed);
            }
            let mut s = next.unwrap_or(t + jitter * pitch);
            while found.is_none() && s < stop {
                if inside(s) {
                    found = Some(s);
                    break;
                }
                outside = s;
                s += spacing * ray.footprint(s);
            }
            if found.is_some() && !node.occupied {
                node.struck.store(true, Ordering::Relaxed);
            }
            let Some(s) = found else {
                if exit <= leave {
                    node.through.fetch_add(1, Ordering::Relaxed);
                }
                next = Some(s);
                t = exit.max(t + nudge);
                continue;
            };

            // Bracket the surface between a point outside and `s`. When
            // the steps before `s` were skipped, the point a pitch back
            // has to be checked, and if it is inside too the surface was
            // crossed unseen in skipped space: back off in doubling
            // strides until outside again.
            let pitch = spacing * ray.footprint(s);
            let (mut a, mut b) = (outside, s);
            if s - outside > pitch * 1.001 {
                let mut stride = pitch;
                loop {
                    let back = s - stride;
                    if back <= enter {
                        a = enter;
                        break;
                    }
                    if !inside(back) {
                        a = back;
                        break;
                    }
                    b = back;
                    stride *= 2.0;
                }
            }
            for _ in 0..BISECTIONS {
                let mid = 0.5 * (a + b);
                if inside(mid) {
                    b = mid;
                } else {
                    a = mid;
                }
            }
            let t_hit = 0.5 * (a + b);
            let position = ray.at(t_hit);
            let pitch = spacing * ray.footprint(t_hit);

            // A point the tree already has here needs no normal found.
            if let Some(known) = self.surfel_near(position, pitch, ray.dir) {
                let hit = RayHit {
                    t: t_hit,
                    position,
                    normal: known.normal.into(),
                };
                return Some((hit, None));
            }

            // The normal is that of the plane through this point and two
            // more, found half a pitch to either side in the image.
            let side = (view.right - ray.dir * view.right.dot(ray.dir)).normalize();
            let lift = ray.dir.cross(side);
            let mut neighbour = |offset: DVec3| -> Option<DVec3> {
                let origin = ray.origin + offset * (0.5 * pitch);
                let mut inside = |t: f64| {
                    *samples += 1;
                    let p = origin + ray.dir * t;
                    volumetric_abi::is_occupied(sampler.sample(p.x, p.y, p.z))
                };
                let mut reach = pitch;
                for _ in 0..4 {
                    let (mut a, mut b) = (t_hit - reach, t_hit + reach);
                    if !inside(a) && inside(b) {
                        for _ in 0..BISECTIONS {
                            let mid = 0.5 * (a + b);
                            if inside(mid) {
                                b = mid;
                            } else {
                                a = mid;
                            }
                        }
                        return Some(origin + ray.dir * (0.5 * (a + b)));
                    }
                    reach *= 2.0;
                }
                None
            };
            let normal = match (neighbour(side), neighbour(lift)) {
                (Some(p), Some(q)) => {
                    let normal = (p - position).cross(q - position).normalize_or(-ray.dir);
                    if normal.dot(ray.dir) > 0.0 {
                        -normal
                    } else {
                        normal
                    }
                }
                // No surface beside the point: an edge or something thin.
                // Facing the viewer is the least wrong guess.
                _ => -ray.dir,
            }
            .as_vec3();
            let hit = RayHit {
                t: t_hit,
                position,
                normal,
            };
            return Some((
                hit,
                Some(Fresh {
                    position,
                    normal,
                    pitch,
                    t: t_hit,
                }),
            ));
        }
        None
    }

    /// Casts one lattice of rays `spacing` pixels apart. `previous`, the
    /// image of the pass before at the same spacing, is what a
    /// [`PassMode::Chase`] pass adds to. `None` when cancelled, with the
    /// record unchanged.
    pub fn pass(
        &mut self,
        sampler: &(impl ParallelModelSampler + ?Sized),
        view: &CastView,
        spacing: f64,
        mode: PassMode,
        previous: Option<CastImage>,
        cancel: &AtomicBool,
    ) -> Option<(CastImage, PassStats)> {
        let start = web_time::Instant::now();
        let (width, height) = CastImage::lattice(view, spacing);
        let previous = previous.filter(|image| {
            mode == PassMode::Chase && (image.width, image.height) == (width, height)
        });
        let salt = self.passes;
        let this = &*self;
        let previous_ref = previous.as_ref();
        let rows = crate::parallel_iter::map_range(0..height as usize, |row| {
            let mut hits = Vec::with_capacity(width as usize);
            let mut fresh = Vec::new();
            let mut samples = 0u64;
            for column in 0..width as usize {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let before =
                    previous_ref.and_then(|image| image.hits[row * width as usize + column]);
                let ray = view.ray(
                    (column as f64 + 0.5) * spacing,
                    (row as f64 + 0.5) * spacing,
                );
                let limit = before.map_or(f64::INFINITY, |hit| hit.t);
                let jitter = unit_hash(column as u32, row as u32, salt);
                let traced = this.trace(
                    sampler,
                    view,
                    &ray,
                    spacing,
                    mode,
                    jitter,
                    limit,
                    &mut samples,
                );
                hits.push(match traced {
                    Some((hit, found)) => {
                        fresh.extend(found.map(|found| (column, found)));
                        Some(hit)
                    }
                    None => before,
                });
            }
            (hits, fresh, samples)
        });
        if cancel.load(Ordering::Relaxed) {
            self.clear_through();
            return None;
        }

        let mut stats = PassStats {
            rays: width as u64 * height as u64,
            ..Default::default()
        };
        let mut image = CastImage {
            width,
            height,
            spacing,
            hits: Vec::with_capacity(width as usize * height as usize),
        };
        let mut found = Vec::new();
        for (row, (hits, fresh, samples)) in rows.into_iter().enumerate() {
            stats.samples += samples;
            stats.hits += hits.iter().flatten().count() as u64;
            image.hits.extend(hits);
            found.extend(
                fresh
                    .into_iter()
                    .map(|(column, fresh)| (column, row, fresh)),
            );
        }
        for (column, row, fresh) in &found {
            // Surface is followed into the nodes around a hit only where
            // the image says it is thin or its extent unknown: where a ray
            // beside this one missed, or hit at quite another depth. A
            // chase is already following, and goes on.
            let beside = |dx: i32, dy: i32| {
                let (x, y) = (*column as i32 + dx, *row as i32 + dy);
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    return true;
                }
                image
                    .hit(x as u32, y as u32)
                    .is_some_and(|hit| (hit.t - fresh.t).abs() <= ISOLATED_DEPTH * fresh.pitch)
            };
            let follow = mode == PassMode::Chase
                || !(beside(-1, 0) && beside(1, 0) && beside(0, -1) && beside(0, 1));
            if self.insert(fresh, follow) {
                stats.fresh += 1;
                stats.followed += follow as u64;
            }
        }
        self.settle(view, spacing);
        self.passes += 1;
        stats.seconds = start.elapsed().as_secs_f64();
        self.total += stats;
        self.history.push((mode, spacing, stats));
        Some((image, stats))
    }

    /// Stores a surface point at the level matching its pitch. `false`
    /// when a point just like it is already there.
    fn insert(&mut self, found: &Fresh, follow: bool) -> bool {
        let level = self.level_for(found.pitch);
        let at = self.ensure(found.position, level);
        let reach = 0.5 * found.pitch;
        let position = found.position.as_vec3();
        let duplicate = self.nodes[at.index as usize].surfels.iter().any(|s| {
            (s.radius as f64) < RADIUS_PER_PITCH * found.pitch * SAME_PITCH
                && Vec3::from(s.position).distance_squared(position) < (reach * reach) as f32
                && Vec3::from(s.normal).dot(found.normal) > 0.5
        });
        if duplicate {
            return false;
        }

        // Mark the way down as holding surface.
        let first_here = self.nodes[at.index as usize].surfels.is_empty();
        let mut walk = self.root();
        for _ in 0..=level {
            let node = &mut self.nodes[walk.index as usize];
            node.occupied = true;
            if walk.index == at.index {
                break;
            }
            let first_child = node.first_child;
            let (index, center, half) = Self::child_of(&walk, first_child, found.position);
            walk = Located {
                index,
                center,
                half,
                ..walk
            };
        }
        self.nodes[at.index as usize].surfels.push(Surfel {
            position: position.into(),
            normal: found.normal.into(),
            radius: (RADIUS_PER_PITCH * found.pitch) as f32,
        });

        // Surface found here may continue into the nodes around, whatever
        // a coarser search of them concluded.
        if follow && first_here {
            let pitch = Self::pitch_of(at.half) as f32;
            for offset in neighbour_offsets() {
                let point = at.center + offset * (2.0 * at.half);
                if (point - self.center).abs().max_element() >= self.half {
                    continue;
                }
                let beside = self.ensure(point, level);
                let node = &mut self.nodes[beside.index as usize];
                if !node.occupied && beside.searched > pitch {
                    node.suspect = true;
                }
            }
        }
        true
    }

    fn clear_through(&mut self) {
        for node in &mut self.nodes {
            *node.through.get_mut() = 0;
            *node.reached.get_mut() = false;
            *node.struck.get_mut() = false;
        }
    }

    /// After a pass: every empty node enough rays went through is now
    /// searched at the pass's pitch there.
    fn settle(&mut self, view: &CastView, spacing: f64) {
        // A node a sample was inside the model in holds matter, even when
        // bisection put the surface point itself in the node before it.
        // Marking it keeps it from counting as searched and empty, and
        // has later rays step through it.
        let mut struck = Vec::new();
        let mut stack = vec![self.root()];
        while let Some(at) = stack.pop() {
            let node = &mut self.nodes[at.index as usize];
            if std::mem::take(node.struck.get_mut()) {
                struck.push(at.center);
            }
            if node.first_child != 0 {
                stack.extend(at.children(node.first_child));
            }
        }
        for center in struck {
            let mut walk = self.root();
            loop {
                let node = &mut self.nodes[walk.index as usize];
                node.occupied = true;
                if node.first_child == 0 || walk.center == center {
                    break;
                }
                let (index, center, half) = Self::child_of(&walk, node.first_child, center);
                walk = Located {
                    index,
                    center,
                    half,
                    ..walk
                };
            }
        }

        let mut stack = vec![self.root()];
        while let Some(at) = stack.pop() {
            let node = &mut self.nodes[at.index as usize];
            let through = std::mem::take(node.through.get_mut());
            // A suspect node gets one look. If the rays that reached it
            // found nothing, the rest of it is left to the searches.
            if std::mem::take(node.reached.get_mut()) {
                node.suspect = false;
            }
            if through > 0 && !node.occupied {
                // The part of the node inside the model's bounds is all
                // the rays could cross.
                let low = (at.center - at.half).max(self.bounds_min);
                let high = (at.center + at.half).min(self.bounds_max);
                let size = (high - low).max(DVec3::ZERO);
                let dir = view.direction_at(at.center).abs();
                let outline =
                    dir.x * size.y * size.z + dir.y * size.x * size.z + dir.z * size.x * size.y;
                let pitch = spacing * view.footprint_at(at.center);
                if through as f64 >= COVERAGE * outline / (pitch * pitch) {
                    node.searched = node.searched.min(pitch as f32);
                }
            }
            let first_child = node.first_child;
            if first_child != 0 {
                stack.extend(at.children(first_child));
            }
        }
    }

    /// Runs one pass at the finished image's spacing, then
    /// [`PassMode::Chase`] passes until one finds nothing more to follow,
    /// adding what they cost to `total`.
    fn pass_and_chase(
        &mut self,
        sampler: &(impl ParallelModelSampler + ?Sized),
        view: &CastView,
        spacing: f64,
        mode: PassMode,
        cancel: &AtomicBool,
        total: &mut PassStats,
    ) -> Option<CastImage> {
        let (mut image, mut stats) = self.pass(sampler, view, spacing, mode, None, cancel)?;
        *total += stats;
        for _ in 0..MAX_CHASES {
            if stats.followed == 0 {
                break;
            }
            (image, stats) =
                self.pass(sampler, view, spacing, PassMode::Chase, Some(image), cancel)?;
            *total += stats;
        }
        Some(image)
    }

    /// Takes `view` from whatever the record already holds to a finished
    /// image. `None` when cancelled.
    ///
    /// The order is the one a viewport wants: a coarse search, then known
    /// surface redrawn at each finer spacing down to the final one, where
    /// thin things the search happened on are followed along their
    /// length. Only then is empty space searched at each finer spacing
    /// down to `options.search`, which is where the time goes; whatever
    /// that finds is redrawn and followed the same way.
    pub fn cast(
        &mut self,
        sampler: &(impl ParallelModelSampler + ?Sized),
        view: &CastView,
        options: &CastOptions,
        cancel: &AtomicBool,
    ) -> Option<(CastImage, PassStats)> {
        let mut total = PassStats::default();
        let mut levels = vec![options.spacing];
        while levels[levels.len() - 1] * 2.0 <= options.coarsest {
            levels.push(levels[levels.len() - 1] * 2.0);
        }
        levels.reverse();
        let last = levels.len() - 1;

        let mut image = None;
        for (i, &spacing) in levels.iter().enumerate() {
            let mode = if i == 0 {
                PassMode::Discover
            } else {
                PassMode::Refine
            };
            if i == last {
                image =
                    Some(self.pass_and_chase(sampler, view, spacing, mode, cancel, &mut total)?);
            } else {
                total += self.pass(sampler, view, spacing, mode, None, cancel)?.1;
            }
        }
        for (i, &spacing) in levels.iter().enumerate().skip(1) {
            if spacing < options.search {
                break;
            }
            if i == last {
                image = Some(self.pass_and_chase(
                    sampler,
                    view,
                    spacing,
                    PassMode::Discover,
                    cancel,
                    &mut total,
                )?);
                continue;
            }
            let (_, stats) = self.pass(sampler, view, spacing, PassMode::Discover, None, cancel)?;
            total += stats;
            if stats.fresh > 0 {
                image = Some(self.pass_and_chase(
                    sampler,
                    view,
                    options.spacing,
                    PassMode::Refine,
                    cancel,
                    &mut total,
                )?);
            }
        }
        Some((image?, total))
    }
}

/// Most [`PassMode::Chase`] passes run after one pass. Each follows found
/// surface one node further, so this bounds how far something thin is
/// followed from where it was first hit: this many nodes, each
/// [`CELLS_PER_NODE`] pixels wide.
const MAX_CHASES: usize = 64;

fn neighbour_offsets() -> impl Iterator<Item = DVec3> {
    (0..27).filter(|&i| i != 13).map(|i| {
        DVec3::new(
            (i % 3) as f64 - 1.0,
            (i / 3 % 3) as f64 - 1.0,
            (i / 9) as f64 - 1.0,
        )
    })
}

/// A number in [0, 1) that is the same for the same ray of the same pass.
fn unit_hash(x: u32, y: u32, salt: u32) -> f64 {
    let mut h = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(salt.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    h as f64 / (u32::MAX as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{ModelBounds, WasmBackendError};

    /// A model given as a closure, inside the cube of half-side `extent`.
    struct Analytic<F> {
        extent: f64,
        inside: F,
    }

    impl<F: Fn(DVec3) -> bool + Send + Sync> ParallelModelSampler for Analytic<F> {
        fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
            (self.inside)(DVec3::new(x, y, z)) as u8 as f32
        }

        fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError> {
            let e = self.extent;
            Ok(ModelBounds::new((-e, -e, -e), (e, e, e)))
        }
    }

    static NEVER: AtomicBool = AtomicBool::new(false);

    fn sphere() -> Analytic<impl Fn(DVec3) -> bool + Send + Sync> {
        Analytic {
            extent: 1.0,
            inside: |p: DVec3| p.length() <= 0.8,
        }
    }

    /// Signed distance to a torus about the z axis.
    fn torus_distance(p: DVec3) -> f64 {
        let ring = (p.x * p.x + p.y * p.y).sqrt() - 0.6;
        (ring * ring + p.z * p.z).sqrt() - 0.2
    }

    fn views(size: u32) -> [CastView; 2] {
        let eye = DVec3::new(2.2, -2.9, 1.7);
        [
            CastView::look_at(
                eye,
                DVec3::ZERO,
                DVec3::Z,
                CastProjection::Perspective {
                    tan_half_fov_y: 0.35,
                },
                size,
                size,
            ),
            CastView::look_at(
                eye,
                DVec3::ZERO,
                DVec3::Z,
                CastProjection::Orthographic { half_height: 1.1 },
                size,
                size,
            ),
        ]
    }

    /// Largest distance of a hit from the surface (in pixels of the view
    /// there) and largest angle of a normal from the true one (degrees),
    /// over hits at least `margin` pixels in from any outline.
    fn errors(
        image: &CastImage,
        view: &CastView,
        distance: impl Fn(DVec3) -> f64,
        normal: impl Fn(DVec3) -> DVec3,
        margin: i32,
    ) -> (f64, f64, usize) {
        let (mut worst_distance, mut worst_angle, mut count) = (0.0f64, 0.0f64, 0);
        for y in 0..image.height as i32 {
            for x in 0..image.width as i32 {
                let Some(hit) = image.hit(x as u32, y as u32) else {
                    continue;
                };
                // Interior: every ray around hit too, at much the same
                // depth, so no outline of the model passes nearby.
                let pixel = view.footprint_at(hit.position);
                let interior = (-margin..=margin).all(|dy| {
                    (-margin..=margin).all(|dx| {
                        let (x, y) = (x + dx, y + dy);
                        x >= 0
                            && y >= 0
                            && x < image.width as i32
                            && y < image.height as i32
                            && image
                                .hit(x as u32, y as u32)
                                .is_some_and(|near| (near.t - hit.t).abs() < 12.0 * pixel)
                    })
                });
                let off = distance(hit.position).abs() / view.footprint_at(hit.position);
                worst_distance = worst_distance.max(off);
                if interior {
                    let angle = hit
                        .normal
                        .as_dvec3()
                        .angle_between(normal(hit.position))
                        .to_degrees();
                    worst_angle = worst_angle.max(angle);
                    count += 1;
                }
            }
        }
        (worst_distance, worst_angle, count)
    }

    /// Every hit lies on the sphere to a small fraction of a pixel, the
    /// normals away from the silhouette are right to a degree, and the
    /// image has the sphere's outline.
    #[test]
    fn a_sphere_is_found_exactly() {
        let model = sphere();
        for view in views(160) {
            let mut cast = DirectCast::for_model(&model).unwrap();
            let (image, stats) = cast
                .cast(&model, &view, &CastOptions::default(), &NEVER)
                .unwrap();
            let (distance, angle, interior) =
                errors(&image, &view, |p| p.length() - 0.8, |p| p.normalize(), 2);
            assert!(distance < 0.01, "{distance} px off the surface");
            // The three points of a normal are half a pixel apart in the
            // image, which near the silhouette is a long way round the
            // sphere: the plane through them leans a couple of degrees.
            assert!(angle < 2.5, "{angle} degrees off the normal");
            let (_, angle, _) = errors(&image, &view, |p| p.length() - 0.8, |p| p.normalize(), 16);
            assert!(angle < 1.0, "{angle} degrees off the normal well inside");

            // The outline: a ray hits exactly when it passes within the
            // radius of the centre, give or take a pixel at the edge.
            let mut wrong = 0;
            for y in 0..image.height {
                for x in 0..image.width {
                    let ray = view.ray(x as f64 + 0.5, y as f64 + 0.5);
                    let closest = ray.at(-ray.origin.dot(ray.dir));
                    let miss_by = (closest.length() - 0.8) / view.footprint_at(closest);
                    if miss_by.abs() > 1.0 && (miss_by < 0.0) != image.hit(x, y).is_some() {
                        wrong += 1;
                    }
                }
            }
            assert_eq!(wrong, 0, "pixels on the wrong side of the outline");
            assert!(interior > 5_000, "{interior} interior hits");
            assert!(stats.samples > 0);
        }
    }

    #[test]
    fn a_torus_is_found_exactly() {
        let model = Analytic {
            extent: 1.0,
            inside: |p: DVec3| torus_distance(p) <= 0.0,
        };
        let gradient = |p: DVec3| {
            let e = 1e-6;
            DVec3::new(
                torus_distance(p + DVec3::X * e) - torus_distance(p - DVec3::X * e),
                torus_distance(p + DVec3::Y * e) - torus_distance(p - DVec3::Y * e),
                torus_distance(p + DVec3::Z * e) - torus_distance(p - DVec3::Z * e),
            )
            .normalize()
        };
        for view in views(160) {
            let mut cast = DirectCast::for_model(&model).unwrap();
            let (image, _) = cast
                .cast(&model, &view, &CastOptions::default(), &NEVER)
                .unwrap();
            // The margin keeps clear of the silhouette and of the line
            // where the near side of the ring crosses in front of the far
            // side: at both, a normal's three points straddle two surfaces.
            let (distance, angle, interior) = errors(&image, &view, torus_distance, gradient, 3);
            assert!(distance < 0.01, "{distance} px off the surface");
            assert!(interior > 2_000, "{interior} interior hits");
            // Curvature across half a pixel bends the three-point plane;
            // on the tight inner ring that is a couple of degrees.
            assert!(angle < 4.0, "{angle} degrees off the normal");
        }
    }

    #[test]
    fn the_same_cast_gives_the_same_result() {
        let model = sphere();
        let view = views(96)[0];
        let run = || {
            let mut cast = DirectCast::for_model(&model).unwrap();
            let (image, stats) = cast
                .cast(&model, &view, &CastOptions::default(), &NEVER)
                .unwrap();
            (image, stats.samples, cast.surfels(None))
        };
        let (a, b) = (run(), run());
        assert!(a == b);
    }

    /// Every surfel sits on the surface: a short way out along its normal
    /// is outside the model, the same way in is inside.
    #[test]
    fn surfels_straddle_the_surface() {
        let model = sphere();
        let view = views(128)[0];
        let mut cast = DirectCast::for_model(&model).unwrap();
        cast.cast(&model, &view, &CastOptions::default(), &NEVER)
            .unwrap();
        let surfels = cast.surfels(None);
        assert!(surfels.len() > 3_000, "{} surfels", surfels.len());
        let mut bad = 0;
        for surfel in &surfels {
            let p = Vec3::from(surfel.position).as_dvec3();
            let n = Vec3::from(surfel.normal).as_dvec3();
            let step = surfel.radius as f64 * 0.2;
            let inside = |p: DVec3| p.length() <= 0.8;
            bad += (inside(p + n * step) || !inside(p - n * step)) as usize;
        }
        // The few that fail took the fallback normal at the silhouette.
        assert!(bad * 50 < surfels.len(), "{bad} of {}", surfels.len());
    }

    /// A grid of rods two pixels thick, which the coarse first search
    /// steps straight over most of. The full search finds all of every
    /// rod; so does the cheap one, which only follows what it happened on.
    #[test]
    fn thin_rods_are_found_whole() {
        let size = 192;
        let view = CastView::look_at(
            DVec3::new(0.4, -3.0, 0.9),
            DVec3::ZERO,
            DVec3::Z,
            CastProjection::Orthographic { half_height: 1.0 },
            size,
            size,
        );
        let radius = view.footprint_at(DVec3::ZERO);
        // Rods along x through a 0.3 lattice in y and z: 25 of them.
        let wrap = |v: f64| (v + 0.15).rem_euclid(0.3) - 0.15;
        let model = Analytic {
            extent: 0.8,
            inside: move |p: DVec3| (wrap(p.y).powi(2) + wrap(p.z).powi(2)).sqrt() <= radius,
        };
        // The pixels a rod certainly covers, with the rod: those whose ray
        // passes within half a radius of a rod's axis.
        let expected: Vec<((u32, u32), (i32, i32))> = (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .filter_map(|(x, y)| {
                let ray = view.ray(x as f64 + 0.5, y as f64 + 0.5);
                (0..2000).find_map(|i| {
                    let p = ray.at(1.5 + i as f64 * 0.0015);
                    let on = p.abs().max_element() < 0.8
                        && (wrap(p.y).powi(2) + wrap(p.z).powi(2)).sqrt() <= radius * 0.5;
                    on.then(|| {
                        (
                            (x, y),
                            ((p.y / 0.3).round() as i32, (p.z / 0.3).round() as i32),
                        )
                    })
                })
            })
            .collect();
        assert!(expected.len() > 2_000, "{} rod pixels", expected.len());

        // For each rod, the fraction of its pixels the image shows.
        let found = |search: f64| {
            let mut cast = DirectCast::for_model(&model).unwrap();
            let options = CastOptions {
                search,
                ..Default::default()
            };
            let (image, stats) = cast.cast(&model, &view, &options, &NEVER).unwrap();
            let mut rods = std::collections::BTreeMap::<(i32, i32), (f64, f64)>::new();
            for &((x, y), rod) in &expected {
                let entry = rods.entry(rod).or_default();
                entry.0 += image.hit(x, y).is_some() as u8 as f64;
                entry.1 += 1.0;
            }
            let rods: Vec<f64> = rods.values().map(|(hit, all)| hit / all).collect();
            (rods, stats.samples)
        };
        let (full, full_samples) = found(1.0);
        let (cheap, cheap_samples) = found(f64::INFINITY);
        let whole = cheap.iter().filter(|&&rod| rod > 0.97).count();
        println!("full search: {full_samples} samples, rods {full:.2?}");
        println!("coarse search and following: {cheap_samples} samples, rods {cheap:.2?}");
        assert!(full.len() >= 20, "{} rods in view", full.len());
        assert!(
            full.iter().all(|&rod| rod > 0.99),
            "the full search left part of a rod unfound"
        );
        // Following cannot find a rod the coarse search never touched,
        // but one it touched it finds all of.
        assert!(
            cheap.iter().all(|&rod| rod == 0.0 || rod > 0.97),
            "following left a rod part found"
        );
        assert!(whole * 2 > cheap.len(), "only {whole} rods followed");
        assert!(cheap_samples * 2 < full_samples);
    }

    /// A model that fills its bounds is hit where rays enter them, with
    /// the normal of the face entered.
    #[test]
    fn a_model_filling_its_bounds_is_hit_at_them() {
        let model = Analytic {
            extent: 0.5,
            inside: |_: DVec3| true,
        };
        let view = views(64)[0];
        let mut cast = DirectCast::for_model(&model).unwrap();
        let (image, _) = cast
            .cast(&model, &view, &CastOptions::default(), &NEVER)
            .unwrap();
        let hits: Vec<&RayHit> = image.hits.iter().flatten().collect();
        assert!(hits.len() > 300, "{} hits", hits.len());
        for hit in hits {
            let face = hit.position.abs().max_position();
            assert!((hit.position[face].abs() - 0.5).abs() < 1e-9);
            let mut normal = DVec3::ZERO;
            normal[face] = hit.position[face].signum();
            assert_eq!(hit.normal.as_dvec3(), normal);
        }
    }

    /// What one view found is not paid for again by the next: a second
    /// view a few degrees round costs far fewer samples than it would
    /// from nothing, and shows the same surface.
    #[test]
    fn a_second_view_reuses_the_first() {
        let model = sphere();
        let [first, _] = views(128);
        let second = CastView::look_at(
            DVec3::new(2.5, -2.6, 1.7),
            DVec3::ZERO,
            DVec3::Z,
            first.projection,
            128,
            128,
        );
        let options = CastOptions::default();

        let mut fresh = DirectCast::for_model(&model).unwrap();
        let (alone, alone_stats) = fresh.cast(&model, &second, &options, &NEVER).unwrap();

        let mut cast = DirectCast::for_model(&model).unwrap();
        cast.cast(&model, &first, &options, &NEVER).unwrap();
        let (after, after_stats) = cast.cast(&model, &second, &options, &NEVER).unwrap();

        println!(
            "second view: {} samples from nothing, {} after the first",
            alone_stats.samples, after_stats.samples
        );
        assert!(after_stats.samples * 3 < alone_stats.samples * 2);
        let differing = alone
            .hits
            .iter()
            .zip(&after.hits)
            .filter(|(a, b)| a.is_some() != b.is_some())
            .count();
        assert!(differing < 40, "{differing} pixels differ in coverage");
    }

    /// Zooming in finds finer surface points where the view now looks,
    /// and the surfels chosen for each view suit its pixels.
    #[test]
    fn zooming_in_refines_what_is_looked_at() {
        let model = sphere();
        let options = CastOptions::default();
        let wide = views(96)[0];
        let close = CastView {
            projection: CastProjection::Perspective {
                tan_half_fov_y: 0.04,
            },
            ..wide
        };
        let mut cast = DirectCast::for_model(&model).unwrap();
        cast.cast(&model, &wide, &options, &NEVER).unwrap();
        // Mean radius of the surfels chosen for a view that fall in it.
        let radius = |cast: &DirectCast, view: &CastView| {
            let seen: Vec<f64> = cast
                .surfels(Some((view, 1.0)))
                .iter()
                .filter(|s| view.pixel_of(Vec3::from(s.position).as_dvec3()).is_some())
                .map(|s| s.radius as f64)
                .collect();
            assert!(seen.len() > 1_000, "{} surfels in view", seen.len());
            seen.iter().sum::<f64>() / seen.len() as f64
        };
        let wide_radius = radius(&cast, &wide);
        cast.cast(&model, &close, &options, &NEVER).unwrap();
        let close_radius = radius(&cast, &close);
        let wide_again = radius(&cast, &wide);

        assert!(
            close_radius * 6.0 < wide_radius,
            "{close_radius} against {wide_radius}"
        );
        // The wide view still draws mostly its own coarse surfels.
        assert!(wide_again > wide_radius * 0.5);
    }

    #[test]
    fn a_cancelled_pass_returns_nothing() {
        let model = sphere();
        let mut cast = DirectCast::for_model(&model).unwrap();
        let cancelled = AtomicBool::new(true);
        assert!(
            cast.cast(&model, &views(64)[0], &CastOptions::default(), &cancelled)
                .is_none()
        );
        assert!(cast.surfels(None).is_empty());
    }
}
