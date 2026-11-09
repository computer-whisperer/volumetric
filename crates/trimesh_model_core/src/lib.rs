//! The triangle-mesh model payload: the data contract between
//! `mesh_to_model_operator` (which builds it) and `trimesh_model_template`
//! (which traverses it at sample time).
//!
//! The operator does all the work up front — BVH construction, texture
//! compression and serialization happen at conversion time in
//! [`build_payload`] — so the generated model is stateless: [`PayloadView`]
//! only reads. Both sides are this one crate, natively unit-tested, so the
//! layout can't drift the way hand-emitted codegen offsets did in the old
//! STL importer.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (64 bytes):
//!    0  magic          u32   "TRM2" (0x324D_5254)
//!    4  triangle_count u32
//!    8  node_count     u32
//!   12  payload_len    u32   total byte length, header included
//!   16  bounds         6xf64 [min_x, max_x, min_y, max_y, min_z, max_z]
//! Nodes (node_count x 32 bytes at offset 64):
//!    0  aabb_min  3xf32
//!   12  aabb_max  3xf32
//!   24  a         u32    internal: left child index
//!                        leaf: 0x8000_0000 | first triangle index
//!   28  b         u32    internal: right child index; leaf: triangle count
//! Triangles (triangle_count x 36 bytes after the nodes): 9xf32 v0 v1 v2,
//! reordered so each leaf's triangles are contiguous.
//! Colour section (optional: present iff payload_len extends past the
//! triangles; 16-byte header):
//!    0  kind     u32   1 = per-corner sRGB, 2 = per-corner UV + texture
//!    4  width    u32   texture width in texels (kind 2; 0 otherwise)
//!    8  height   u32   texture height (kind 2; 0 otherwise)
//!   12  format   u32   texel encoding, `TextureFormat` (kind 2; 0 otherwise)
//!   16  corner attributes in the reordered triangle order, 3 corners per
//!       triangle: kind 1: 3xf32 r g b per corner (sRGB, 0..1);
//!                 kind 2: 2xf32 u v per corner (OBJ convention)
//!   then, for kind 2, the texture: Rgb8 = width*height*3 bytes, top row
//!   first; Bc1 = ceil(width/4)*ceil(height/4) blocks of 8 bytes.
//! ```
//!
//! # Inside/outside semantics
//!
//! [`PayloadView::is_inside`] casts a ray in a fixed near-+x direction and
//! classifies by crossing parity. For a watertight mesh this is the usual
//! point-in-solid test. Open meshes get parity's literal behavior — a
//! point is "inside" when an odd number of triangles lie between it and
//! infinity along the ray — which is well-defined but only meaningful for
//! closed surfaces; the conversion operator documents this rather than
//! rejecting open input. The ray is deliberately skewed off the x axis
//! (by irrational-ish factors) so the axis-aligned edges and face
//! diagonals that grid samplers constantly line up with never produce
//! parity ties; points exactly on a surface may still classify either way
//! (no exact arithmetic).
//!
//! # Surface colour
//!
//! [`PayloadView::nearest_color`] is the colour of the closest surface
//! point, defined everywhere in space (so a sampler slightly off a face,
//! or deep inside the solid, still reads the nearest face's colour).
//! Per-corner attributes are interpolated barycentrically at the closest
//! point; a UV lands in the texture with bilinear filtering and repeat
//! wrapping, `v` flipped because images are stored top row first.

pub mod bc1;
pub mod texture;

pub use texture::{TextureFormat, TextureImage};
use volumetric_abi::trimesh::{COLOR_FIELD_NAME, MeshTexture, TriMesh};

pub const MAGIC: u32 = 0x324D_5254; // "TRM2"
const HEADER_LEN: usize = 64;
const NODE_LEN: usize = 32;
const TRI_LEN: usize = 36;
const COLOR_HEADER_LEN: usize = 16;
const LEAF_FLAG: u32 = 0x8000_0000;
const MAX_LEAF_TRIS: usize = 4;
const COLOR_KIND_RGB: u32 = 1;
const COLOR_KIND_UV: u32 = 2;

/// How [`build_payload`] handles a mesh's colour data.
///
/// Preference order: the base-colour texture (when the mesh has one and a
/// decoder is supplied), then the `color` vertex field, then no colour
/// section at all.
#[derive(Default)]
pub struct BuildOptions<'a> {
    /// Texel encoding for a texture section.
    pub texture_format: TextureFormat,
    /// Box-filter the texture down until both extents are at most this
    /// many texels; `0` keeps the full image.
    pub texture_max_size: u32,
    /// Decodes a mesh texture's file bytes (JPEG/PNG...) to pixels. `None`
    /// means textures are not used at all, so a textured mesh falls back to
    /// its vertex colour (if any). A decoder that returns `None` for a
    /// texture (undecodable bytes) gets the same fallback: the geometry is
    /// still a valid solid, so a broken image is the decoder's warning to
    /// post, not a conversion failure.
    #[allow(clippy::type_complexity)]
    pub decode_texture: Option<&'a dyn Fn(&MeshTexture) -> Option<TextureImage>>,
}

/// Build the payload for a mesh: BVH + reordered triangles + optional
/// colour section, serialized.
///
/// Fails on an empty mesh (a solid with no surface is meaningless), on
/// invalid meshes, and when a decoded texture is structurally wrong.
pub fn build_payload(mesh: &TriMesh, options: &BuildOptions) -> Result<Vec<u8>, String> {
    mesh.validate()?;
    let triangle_count = mesh.triangle_count();
    if triangle_count == 0 {
        return Err("mesh has no triangles".to_string());
    }

    // f32 triangle copies (the sample-time precision) and their bounds.
    let tri = |t: usize| -> [[f32; 3]; 3] {
        let idx = mesh.triangle(t);
        std::array::from_fn(|v| {
            let p = mesh.position(idx[v] as usize);
            [p[0] as f32, p[1] as f32, p[2] as f32]
        })
    };

    #[derive(Clone, Copy)]
    struct TriBounds {
        min: [f32; 3],
        max: [f32; 3],
        centroid: [f32; 3],
    }
    let tri_bounds: Vec<TriBounds> = (0..triangle_count)
        .map(|t| {
            let v = tri(t);
            let mut min = v[0];
            let mut max = v[0];
            for p in &v[1..] {
                for axis in 0..3 {
                    min[axis] = min[axis].min(p[axis]);
                    max[axis] = max[axis].max(p[axis]);
                }
            }
            TriBounds {
                min,
                max,
                centroid: std::array::from_fn(|a| (min[a] + max[a]) * 0.5),
            }
        })
        .collect();

    struct Node {
        min: [f32; 3],
        max: [f32; 3],
        a: u32,
        b: u32,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut order: Vec<u32> = (0..triangle_count as u32).collect();

    // Median-split recursion over `order[range]`; leaves reference the
    // final (reordered) triangle positions directly.
    fn build(
        nodes: &mut Vec<Node>,
        order: &mut [u32],
        base: usize,
        tri_bounds: &[TriBounds],
    ) -> u32 {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &t in order.iter() {
            for axis in 0..3 {
                min[axis] = min[axis].min(tri_bounds[t as usize].min[axis]);
                max[axis] = max[axis].max(tri_bounds[t as usize].max[axis]);
            }
        }
        let node_index = nodes.len() as u32;
        nodes.push(Node {
            min,
            max,
            a: 0,
            b: 0,
        });

        if order.len() <= MAX_LEAF_TRIS {
            nodes[node_index as usize].a = LEAF_FLAG | base as u32;
            nodes[node_index as usize].b = order.len() as u32;
            return node_index;
        }

        let extent: [f32; 3] = std::array::from_fn(|a| max[a] - min[a]);
        let axis = (0..3)
            .max_by(|&x, &y| extent[x].partial_cmp(&extent[y]).unwrap())
            .unwrap();
        order.sort_unstable_by(|&p, &q| {
            tri_bounds[p as usize].centroid[axis]
                .partial_cmp(&tri_bounds[q as usize].centroid[axis])
                .unwrap()
        });
        let mid = order.len() / 2;
        let (left, right) = order.split_at_mut(mid);
        let a = build(nodes, left, base, tri_bounds);
        let b = build(nodes, right, base + mid, tri_bounds);
        nodes[node_index as usize].a = a;
        nodes[node_index as usize].b = b;
        node_index
    }
    build(&mut nodes, &mut order, 0, &tri_bounds);

    let bounds = mesh.bounds().expect("non-empty mesh has bounds");

    // Colour source, by preference.
    enum Color {
        Vertex(Vec<f32>),                 // 9 f32 per reordered triangle
        Textured(Vec<f32>, TextureImage), // 6 f32 per reordered triangle
    }
    let color = {
        let texture = match (mesh.base_color_texture(), options.decode_texture) {
            (Some(texture), Some(decode)) => match decode(texture) {
                Some(image) => {
                    image.validate()?;
                    Some(image.shrunk_to(options.texture_max_size))
                }
                None => None,
            },
            _ => None,
        };
        if let Some(image) = texture {
            let uv = mesh
                .uv_field()
                .expect("base_color_texture implies a uv field");
            let mut attrs = Vec::with_capacity(triangle_count * 6);
            for &t in &order {
                for v in mesh.triangle(t as usize) {
                    let o = v as usize * 2;
                    attrs.push(uv.data[o] as f32);
                    attrs.push(uv.data[o + 1] as f32);
                }
            }
            Some(Color::Textured(attrs, image))
        } else if let Some(field) = mesh
            .vertex_fields
            .iter()
            .find(|f| f.name == COLOR_FIELD_NAME && f.components == 3)
        {
            let mut attrs = Vec::with_capacity(triangle_count * 9);
            for &t in &order {
                for v in mesh.triangle(t as usize) {
                    let o = v as usize * 3;
                    attrs.extend(field.data[o..o + 3].iter().map(|&c| c as f32));
                }
            }
            Some(Color::Vertex(attrs))
        } else {
            None
        }
    };

    let geometry_len = HEADER_LEN + nodes.len() * NODE_LEN + triangle_count * TRI_LEN;
    let color_len = match &color {
        None => 0,
        Some(Color::Vertex(attrs)) => COLOR_HEADER_LEN + attrs.len() * 4,
        Some(Color::Textured(attrs, image)) => {
            COLOR_HEADER_LEN
                + attrs.len() * 4
                + match options.texture_format {
                    TextureFormat::Rgb8 => image.rgb.len(),
                    TextureFormat::Bc1 => bc1::encoded_len(image.width, image.height),
                }
        }
    };
    let payload_len = geometry_len + color_len;
    if payload_len > u32::MAX as usize {
        return Err(format!(
            "payload of {payload_len} bytes exceeds the 4 GiB limit"
        ));
    }

    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((triangle_count as u32).to_le_bytes());
    out.extend((nodes.len() as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in bounds {
        out.extend(v.to_le_bytes());
    }
    for node in &nodes {
        for v in node.min {
            out.extend(v.to_le_bytes());
        }
        for v in node.max {
            out.extend(v.to_le_bytes());
        }
        out.extend(node.a.to_le_bytes());
        out.extend(node.b.to_le_bytes());
    }
    for &t in &order {
        for v in tri(t as usize) {
            for c in v {
                out.extend(c.to_le_bytes());
            }
        }
    }
    match color {
        None => {}
        Some(Color::Vertex(attrs)) => {
            out.extend(COLOR_KIND_RGB.to_le_bytes());
            out.extend([0u8; 12]);
            for v in attrs {
                out.extend(v.to_le_bytes());
            }
        }
        Some(Color::Textured(attrs, image)) => {
            out.extend(COLOR_KIND_UV.to_le_bytes());
            out.extend(image.width.to_le_bytes());
            out.extend(image.height.to_le_bytes());
            out.extend((options.texture_format as u32).to_le_bytes());
            for v in attrs {
                out.extend(v.to_le_bytes());
            }
            match options.texture_format {
                TextureFormat::Rgb8 => out.extend(&image.rgb),
                TextureFormat::Bc1 => {
                    out.extend(bc1::encode(image.width, image.height, &image.rgb))
                }
            }
        }
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// The parsed colour-section header.
#[derive(Clone, Copy)]
struct ColorSection {
    kind: u32,
    width: usize,
    height: usize,
    format: TextureFormat,
    /// Byte offset of the per-corner attributes.
    attrs: usize,
    /// Byte offset of the texture data (kind 2).
    texels: usize,
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    triangle_count: usize,
    node_count: usize,
    color: Option<ColorSection>,
}

impl<'a> PayloadView<'a> {
    /// Validate the header and structural sizes.
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("payload shorter than header");
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let triangle_count = u32_at(4) as usize;
        let node_count = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        let geometry_len = HEADER_LEN + node_count * NODE_LEN + triangle_count * TRI_LEN;
        if payload_len < geometry_len || bytes.len() < payload_len {
            return Err("payload length mismatch");
        }
        let color = if payload_len == geometry_len {
            None
        } else {
            if payload_len < geometry_len + COLOR_HEADER_LEN {
                return Err("truncated colour header");
            }
            let kind = u32_at(geometry_len);
            let width = u32_at(geometry_len + 4) as usize;
            let height = u32_at(geometry_len + 8) as usize;
            let format = TextureFormat::from_u32(u32_at(geometry_len + 12))
                .ok_or("unknown texture format")?;
            let attrs = geometry_len + COLOR_HEADER_LEN;
            let (texels, expected) = match kind {
                COLOR_KIND_RGB => {
                    let end = attrs + triangle_count * 9 * 4;
                    (end, end)
                }
                COLOR_KIND_UV => {
                    if width == 0 || height == 0 {
                        return Err("empty texture");
                    }
                    let texels = attrs + triangle_count * 6 * 4;
                    let len = match format {
                        TextureFormat::Rgb8 => width * height * 3,
                        TextureFormat::Bc1 => bc1::encoded_len(width as u32, height as u32),
                    };
                    (texels, texels + len)
                }
                _ => return Err("unknown colour kind"),
            };
            if expected != payload_len {
                return Err("colour section length mismatch");
            }
            Some(ColorSection {
                kind,
                width,
                height,
                format,
                attrs,
                texels,
            })
        };
        Ok(Self {
            bytes,
            triangle_count,
            node_count,
            color,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.triangle_count
    }

    /// Whether the payload carries a colour section (so
    /// [`Self::nearest_color`] returns `Some`).
    pub fn has_color(&self) -> bool {
        self.color.is_some()
    }

    /// `[min_x, max_x, min_y, max_y, min_z, max_z]`.
    pub fn bounds(&self) -> [f64; 6] {
        std::array::from_fn(|i| {
            f64::from_le_bytes(self.bytes[16 + i * 8..24 + i * 8].try_into().unwrap())
        })
    }

    fn f32_at(&self, off: usize) -> f32 {
        f32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap())
    }

    fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap())
    }

    fn triangle(&self, t: usize) -> [[f32; 3]; 3] {
        let base = HEADER_LEN + self.node_count * NODE_LEN + t * TRI_LEN;
        std::array::from_fn(|v| std::array::from_fn(|c| self.f32_at(base + (v * 3 + c) * 4)))
    }

    /// Height of the mesh surface over a lateral point: the extreme
    /// coordinate along `axis` (max when `top`, min otherwise) where the
    /// axis-aligned line through `lat` crosses a triangle. `lat` holds the
    /// other two coordinates in ascending axis order (axis 2 → (x, y),
    /// axis 1 → (x, z), axis 0 → (y, z)). `None` when the line misses every
    /// triangle. Triangles parallel to the line (vertical walls) don't
    /// define a height and are skipped.
    pub fn height_at(&self, lat: [f64; 2], axis: usize, top: bool) -> Option<f64> {
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            2 => (0, 1),
            _ => panic!("axis out of range: {axis}"),
        };
        let mut origin = [0.0f64; 3];
        origin[u] = lat[0];
        origin[v] = lat[1];
        let mut dir = [0.0f64; 3];
        dir[axis] = 1.0;

        if self.node_count == 0 {
            return None;
        }
        let mut best: Option<f64> = None;
        let mut stack = [0u32; 64];
        let mut top_of_stack = 0usize;
        stack[top_of_stack] = 0;
        top_of_stack += 1;

        while top_of_stack > 0 {
            top_of_stack -= 1;
            let node = stack[top_of_stack] as usize;
            let base = HEADER_LEN + node * NODE_LEN;
            // The line is axis-aligned and unbounded, so the node test is a
            // point-in-rect over the two lateral axes.
            let hit = [u, v].iter().zip(lat).all(|(&a, l)| {
                l >= self.f32_at(base + a * 4) as f64 && l <= self.f32_at(base + 12 + a * 4) as f64
            });
            if !hit {
                continue;
            }
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for t in first..first + b as usize {
                    if let Some(h) = line_hit(self.triangle(t), origin, dir) {
                        best = Some(match best {
                            Some(prev) if top => prev.max(h),
                            Some(prev) => prev.min(h),
                            None => h,
                        });
                    }
                }
            } else {
                stack[top_of_stack] = a;
                top_of_stack += 1;
                stack[top_of_stack] = b;
                top_of_stack += 1;
            }
        }
        best
    }

    /// Point-in-mesh by ray parity along [`RAY_DIR`] (see the module docs
    /// for open-mesh and on-surface semantics).
    pub fn is_inside(&self, p: [f64; 3]) -> bool {
        if self.node_count == 0 {
            return false;
        }

        let mut crossings = 0usize;
        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;
            let node = stack[top] as usize;
            let base = HEADER_LEN + node * NODE_LEN;
            // Slab test; all RAY_DIR components are positive, so no swaps.
            let mut t_min = 0.0f64;
            let mut t_max = f64::INFINITY;
            for axis in 0..3 {
                let inv = 1.0 / RAY_DIR[axis];
                let lo = (self.f32_at(base + axis * 4) as f64 - p[axis]) * inv;
                let hi = (self.f32_at(base + 12 + axis * 4) as f64 - p[axis]) * inv;
                t_min = t_min.max(lo);
                t_max = t_max.min(hi);
            }
            if t_min > t_max {
                continue;
            }
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for t in first..first + b as usize {
                    if ray_crosses(self.triangle(t), p) {
                        crossings += 1;
                    }
                }
            } else {
                // Depth is bounded by ~log2(n) + a slack for uneven splits;
                // 64 covers any mesh whose index fits in u32.
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
            }
        }
        crossings % 2 == 1
    }

    /// Squared distance from `p` to a node's box (zero inside).
    fn node_dist2(&self, node: usize, p: [f64; 3]) -> f64 {
        let base = HEADER_LEN + node * NODE_LEN;
        let mut d2 = 0.0f64;
        for (axis, &c) in p.iter().enumerate() {
            let lo = self.f32_at(base + axis * 4) as f64;
            let hi = self.f32_at(base + 12 + axis * 4) as f64;
            let gap = (lo - c).max(c - hi).max(0.0);
            d2 += gap * gap;
        }
        d2
    }

    /// The closest surface point to `p`: `(triangle, barycentric weights)`
    /// in the reordered triangle numbering. Ordered BVH descent pruned by
    /// the best squared distance so far.
    fn closest_triangle(&self, p: [f64; 3]) -> Option<(usize, [f64; 3])> {
        if self.node_count == 0 {
            return None;
        }
        let mut best_d2 = f64::INFINITY;
        let mut best: Option<(usize, [f64; 3])> = None;
        let mut stack = [(0u32, 0.0f64); 64];
        stack[0] = (0, self.node_dist2(0, p));
        let mut top = 1usize;
        while top > 0 {
            top -= 1;
            let (node, d2) = stack[top];
            if d2 >= best_d2 {
                continue;
            }
            let base = HEADER_LEN + node as usize * NODE_LEN;
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for t in first..first + b as usize {
                    let (q, bary) = closest_point_on_triangle(self.triangle(t), p);
                    let d2 = (0..3).map(|k| (q[k] - p[k]) * (q[k] - p[k])).sum::<f64>();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = Some((t, bary));
                    }
                }
            } else {
                // Push the farther child first so the nearer pops first.
                let da = self.node_dist2(a as usize, p);
                let db = self.node_dist2(b as usize, p);
                let (near, far) = if da <= db {
                    ((a, da), (b, db))
                } else {
                    ((b, db), (a, da))
                };
                if far.1 < best_d2 {
                    stack[top] = far;
                    top += 1;
                }
                if near.1 < best_d2 {
                    stack[top] = near;
                    top += 1;
                }
            }
        }
        best
    }

    /// sRGB colour (0..1) of the surface point nearest `p`, or `None` when
    /// the payload has no colour section. Defined everywhere in space; see
    /// the module docs.
    pub fn nearest_color(&self, p: [f64; 3]) -> Option<[f32; 3]> {
        let color = self.color?;
        let (t, bary) = self.closest_triangle(p)?;
        match color.kind {
            COLOR_KIND_RGB => {
                let base = color.attrs + t * 9 * 4;
                Some(std::array::from_fn(|k| {
                    (0..3)
                        .map(|c| bary[c] * self.f32_at(base + (c * 3 + k) * 4) as f64)
                        .sum::<f64>() as f32
                }))
            }
            _ => {
                let base = color.attrs + t * 6 * 4;
                let uv: [f64; 2] = std::array::from_fn(|k| {
                    (0..3)
                        .map(|c| bary[c] * self.f32_at(base + (c * 2 + k) * 4) as f64)
                        .sum::<f64>()
                });
                Some(self.sample_texture(&color, uv))
            }
        }
    }

    /// Bilinear, repeat-wrapped texture fetch at an OBJ-convention UV.
    fn sample_texture(&self, color: &ColorSection, uv: [f64; 2]) -> [f32; 3] {
        let (w, h) = (color.width as i64, color.height as i64);
        let wrap = |t: f64| if t.is_finite() { t - t.floor() } else { 0.0 };
        // Texel centres sit at integer + 0.5; v counts up from the bottom.
        let x = wrap(uv[0]) * w as f64 - 0.5;
        let y = (1.0 - wrap(uv[1])) * h as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = ((x - x0) as f32, (y - y0) as f32);
        let xi = |x: f64| (x as i64).rem_euclid(w) as usize;
        let yi = |y: f64| (y as i64).rem_euclid(h) as usize;
        let c00 = self.texel(color, xi(x0), yi(y0));
        let c10 = self.texel(color, xi(x0 + 1.0), yi(y0));
        let c01 = self.texel(color, xi(x0), yi(y0 + 1.0));
        let c11 = self.texel(color, xi(x0 + 1.0), yi(y0 + 1.0));
        std::array::from_fn(|k| {
            let top = c00[k] as f32 * (1.0 - tx) + c10[k] as f32 * tx;
            let bottom = c01[k] as f32 * (1.0 - tx) + c11[k] as f32 * tx;
            (top * (1.0 - ty) + bottom * ty) / 255.0
        })
    }

    fn texel(&self, color: &ColorSection, x: usize, y: usize) -> [u8; 3] {
        match color.format {
            TextureFormat::Rgb8 => {
                let o = color.texels + (y * color.width + x) * 3;
                [self.bytes[o], self.bytes[o + 1], self.bytes[o + 2]]
            }
            TextureFormat::Bc1 => {
                let blocks_wide = bc1::blocks_along(color.width as u32);
                let b = (y / 4) * blocks_wide + x / 4;
                let o = color.texels + b * bc1::BLOCK_LEN;
                let block: &[u8; 8] = self.bytes[o..o + 8].try_into().unwrap();
                bc1::decode_block(block)[(y % 4) * 4 + x % 4]
            }
        }
    }
}

/// The parity ray's direction: near +x, skewed so axis-aligned edges and
/// face diagonals never line up with it.
const RAY_DIR: [f64; 3] = [1.0, 1.618_033_988_7e-4, 2.718_281_828_4e-4];

/// Möller–Trumbore for the [`RAY_DIR`] ray from `p`: does the ray cross
/// this triangle at t > 0?
fn ray_crosses(v: [[f32; 3]; 3], p: [f64; 3]) -> bool {
    matches!(line_hit(v, p, RAY_DIR), Some(t) if t > 0.0)
}

/// Möller–Trumbore for the unbounded line `origin + t * dir`: the `t` where
/// the line crosses this triangle, any sign. `None` when it misses or when
/// the triangle is parallel to the line.
fn line_hit(v: [[f32; 3]; 3], origin: [f64; 3], dir: [f64; 3]) -> Option<f64> {
    // All in f64 for headroom; vertex inputs are exact f32 values.
    let (v0, v1, v2) = (v[0], v[1], v[2]);
    let e1 = [
        (v1[0] - v0[0]) as f64,
        (v1[1] - v0[1]) as f64,
        (v1[2] - v0[2]) as f64,
    ];
    let e2 = [
        (v2[0] - v0[0]) as f64,
        (v2[1] - v0[1]) as f64,
        (v2[2] - v0[2]) as f64,
    ];
    let h = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    ];
    let det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if det.abs() < 1e-12 {
        return None; // line parallel to the triangle plane
    }
    let inv_det = 1.0 / det;
    let s = [
        origin[0] - v0[0] as f64,
        origin[1] - v0[1] as f64,
        origin[2] - v0[2] as f64,
    ];
    let u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let w = (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]) * inv_det;
    if w < 0.0 || u + w > 1.0 {
        return None;
    }
    Some((e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_det)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Closest point on triangle `abc` to `p` (Ericson, *Real-Time Collision
/// Detection* 5.1.5) and its barycentric weights `(for a, b, c)`.
fn closest_point_on_triangle(tri: [[f32; 3]; 3], p: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let a = tri[0].map(f64::from);
    let b = tri[1].map(f64::from);
    let c = tri[2].map(f64::from);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let at = |bary: [f64; 3]| -> ([f64; 3], [f64; 3]) {
        (
            std::array::from_fn(|k| bary[0] * a[k] + bary[1] * b[k] + bary[2] * c[k]),
            bary,
        )
    };

    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return at([1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return at([0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom != 0.0 { d1 / denom } else { 0.0 };
        return at([1.0 - v, v, 0.0]);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return at([0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom != 0.0 { d2 / denom } else { 0.0 };
        return at([1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom != 0.0 { (d4 - d3) / denom } else { 0.0 };
        return at([0.0, 1.0 - w, w]);
    }
    let denom = va + vb + vc;
    if denom == 0.0 {
        return at([1.0, 0.0, 0.0]); // degenerate triangle
    }
    let v = vb / denom;
    let w = vc / denom;
    at([1.0 - v - w, v, w])
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::trimesh::{MeshField, UV_FIELD_NAME};

    fn cube(min: f64, max: f64) -> TriMesh {
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { min } else { max },
                    if i & 2 == 0 { min } else { max },
                    if i & 4 == 0 { min } else { max },
                ]
            })
            .collect();
        // Faces as (a, b, c, d) quads with outward CCW winding.
        let quads = [
            [0, 4, 6, 2], // -x
            [1, 3, 7, 5], // +x
            [0, 1, 5, 4], // -y
            [2, 6, 7, 3], // +y
            [0, 2, 3, 1], // -z
            [4, 5, 7, 6], // +z
        ];
        let mut indices = Vec::new();
        for q in quads {
            indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
        TriMesh {
            positions: corners.into_iter().flatten().collect(),
            indices,
            vertex_fields: vec![],
            face_fields: vec![],
            textures: vec![],
        }
    }

    fn plain(mesh: &TriMesh) -> Vec<u8> {
        build_payload(mesh, &BuildOptions::default()).unwrap()
    }

    /// Brute-force parity over all triangles (no BVH) — the oracle.
    fn brute_force_inside(mesh: &TriMesh, p: [f64; 3]) -> bool {
        let mut crossings = 0;
        for t in 0..mesh.triangle_count() {
            let idx = mesh.triangle(t);
            let v: [[f32; 3]; 3] = std::array::from_fn(|i| {
                let q = mesh.position(idx[i] as usize);
                [q[0] as f32, q[1] as f32, q[2] as f32]
            });
            if ray_crosses(v, p) {
                crossings += 1;
            }
        }
        crossings % 2 == 1
    }

    #[test]
    fn cube_payload_classifies_correctly() {
        let payload = plain(&cube(-1.0, 1.0));
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.triangle_count(), 12);
        assert_eq!(view.bounds(), [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        assert!(!view.has_color());
        assert_eq!(view.nearest_color([0.0, 0.0, 0.0]), None);

        assert!(view.is_inside([0.0, 0.0, 0.0]));
        assert!(view.is_inside([0.9, -0.9, 0.9]));
        assert!(!view.is_inside([1.1, 0.0, 0.0]));
        assert!(!view.is_inside([0.0, -1.2, 0.0]));
        assert!(!view.is_inside([5.0, 5.0, 5.0]));
    }

    /// A unit UV sphere with enough triangles to force a real tree.
    fn uv_sphere(rings: usize, segments: usize) -> TriMesh {
        let mut positions: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.0, -1.0];
        for r in 1..rings {
            let phi = std::f64::consts::PI * r as f64 / rings as f64;
            for s in 0..segments {
                let theta = std::f64::consts::TAU * s as f64 / segments as f64;
                positions.extend([phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]);
            }
        }
        let ring = |r: usize, s: usize| (2 + (r - 1) * segments + (s % segments)) as u32;
        let mut indices: Vec<u32> = Vec::new();
        for s in 0..segments {
            indices.extend([0, ring(1, s), ring(1, s + 1)]);
            indices.extend([1, ring(rings - 1, s + 1), ring(rings - 1, s)]);
        }
        for r in 1..rings - 1 {
            for s in 0..segments {
                let (a, b, c, d) = (
                    ring(r, s),
                    ring(r, s + 1),
                    ring(r + 1, s + 1),
                    ring(r + 1, s),
                );
                indices.extend([a, b, c, a, c, d]);
            }
        }
        TriMesh {
            positions,
            indices,
            vertex_fields: vec![],
            face_fields: vec![],
            textures: vec![],
        }
    }

    #[test]
    fn bvh_matches_brute_force_on_a_sphere() {
        let mesh = uv_sphere(16, 24);
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();

        // Deterministic pseudo-random probes, compared against the
        // no-BVH oracle; also sanity-check against the true sphere away
        // from the tessellation skin.
        let mut state = 0x12345678u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 3.0 - 1.5
        };
        for _ in 0..1000 {
            let p = [rand(), rand(), rand()];
            assert_eq!(
                view.is_inside(p),
                brute_force_inside(&mesh, p),
                "BVH disagrees with brute force at {p:?}"
            );
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if r < 0.95 {
                assert!(view.is_inside(p), "sphere interior misclassified at {p:?}");
            }
            if r > 1.05 {
                assert!(!view.is_inside(p), "sphere exterior misclassified at {p:?}");
            }
        }
    }

    #[test]
    fn cube_height_query_hits_the_faces() {
        let payload = plain(&cube(-1.0, 1.0));
        let view = PayloadView::new(&payload).unwrap();

        // Top/bottom along each axis; the lateral point (0.3, -0.2) is well
        // inside every face.
        for axis in 0..3 {
            assert_eq!(view.height_at([0.3, -0.2], axis, true), Some(1.0));
            assert_eq!(view.height_at([0.3, -0.2], axis, false), Some(-1.0));
        }
        // Outside the footprint the line misses.
        assert_eq!(view.height_at([1.5, 0.0], 2, true), None);
        assert_eq!(view.height_at([0.0, -1.5], 0, true), None);
    }

    #[test]
    fn sphere_height_query_matches_the_analytic_surface() {
        let mesh = uv_sphere(16, 24);
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();

        // Inscribed tessellation: heights sit slightly under sqrt(1 - r^2).
        for &(x, y) in &[(0.0, 0.0), (0.3, 0.1), (-0.5, 0.4), (0.2, -0.6)] {
            let analytic = (1.0f64 - x * x - y * y).sqrt();
            let top = view.height_at([x, y], 2, true).expect("line hits sphere");
            let bottom = view.height_at([x, y], 2, false).expect("line hits sphere");
            assert!(
                (top - analytic).abs() < 0.05,
                "top at ({x}, {y}): {top} vs {analytic}"
            );
            assert!(
                (bottom + analytic).abs() < 0.05,
                "bottom at ({x}, {y}): {bottom} vs {analytic}"
            );
        }
        assert_eq!(view.height_at([1.2, 0.0], 2, true), None);
    }

    #[test]
    fn height_query_matches_brute_force() {
        let mesh = uv_sphere(16, 24);
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();

        // No-BVH oracle: fold line_hit over every triangle.
        let brute = |lat: [f64; 2], axis: usize, top: bool| -> Option<f64> {
            let (u, v) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let mut origin = [0.0f64; 3];
            origin[u] = lat[0];
            origin[v] = lat[1];
            let mut dir = [0.0f64; 3];
            dir[axis] = 1.0;
            let mut best: Option<f64> = None;
            for t in 0..mesh.triangle_count() {
                let idx = mesh.triangle(t);
                let tri: [[f32; 3]; 3] = std::array::from_fn(|i| {
                    let q = mesh.position(idx[i] as usize);
                    [q[0] as f32, q[1] as f32, q[2] as f32]
                });
                if let Some(h) = line_hit(tri, origin, dir) {
                    best = Some(match best {
                        Some(prev) if top => prev.max(h),
                        Some(prev) => prev.min(h),
                        None => h,
                    });
                }
            }
            best
        };

        let mut state = 0xdeadbeefu64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 3.0 - 1.5
        };
        for _ in 0..500 {
            let lat = [rand(), rand()];
            for axis in 0..3 {
                for top in [true, false] {
                    assert_eq!(
                        view.height_at(lat, axis, top),
                        brute(lat, axis, top),
                        "BVH height disagrees with brute force at {lat:?} axis {axis} top {top}"
                    );
                }
            }
        }
    }

    #[test]
    fn open_mesh_has_documented_parity_semantics() {
        // A single triangle in the x = 1 plane: points with the triangle
        // between them and +infinity along x count one crossing -> "inside".
        let mesh = TriMesh {
            positions: vec![
                1.0, -1.0, -1.0, //
                1.0, 1.0, -1.0, //
                1.0, 0.0, 1.0,
            ],
            indices: vec![0, 1, 2],
            vertex_fields: vec![],
            face_fields: vec![],
            textures: vec![],
        };
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.is_inside([0.0, 0.0, 0.0]), "behind the triangle");
        assert!(!view.is_inside([2.0, 0.0, 0.0]), "past the triangle");
        assert!(!view.is_inside([0.0, 5.0, 0.0]), "outside the silhouette");
    }

    #[test]
    fn empty_and_garbage_payloads_are_rejected() {
        let empty = TriMesh {
            positions: vec![],
            indices: vec![],
            vertex_fields: vec![],
            face_fields: vec![],
            textures: vec![],
        };
        assert!(build_payload(&empty, &BuildOptions::default()).is_err());
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mut bad = plain(&cube(0.0, 1.0));
        bad[0] ^= 0xFF; // corrupt magic
        assert!(PayloadView::new(&bad).is_err());

        // A truncated colour section is caught by the length check.
        let mut colored = cube(0.0, 1.0);
        colored.vertex_fields.push(MeshField {
            name: COLOR_FIELD_NAME.to_string(),
            components: 3,
            data: vec![0.5; 24],
        });
        let mut payload = plain(&colored);
        assert!(PayloadView::new(&payload).unwrap().has_color());
        payload.truncate(payload.len() - 4);
        assert!(PayloadView::new(&payload).is_err());
    }

    /// The nearest-colour query's oracle: brute-force closest point over
    /// every triangle, with the mesh's own attribute interpolation.
    fn brute_force_closest(mesh: &TriMesh, p: [f64; 3]) -> (usize, [f64; 3]) {
        let mut best = (f64::INFINITY, 0usize, [0.0; 3]);
        for t in 0..mesh.triangle_count() {
            let idx = mesh.triangle(t);
            let tri: [[f32; 3]; 3] = std::array::from_fn(|i| {
                let q = mesh.position(idx[i] as usize);
                [q[0] as f32, q[1] as f32, q[2] as f32]
            });
            let (q, bary) = closest_point_on_triangle(tri, p);
            let d2 = (0..3).map(|k| (q[k] - p[k]) * (q[k] - p[k])).sum::<f64>();
            if d2 < best.0 {
                best = (d2, t, bary);
            }
        }
        (best.1, best.2)
    }

    #[test]
    fn closest_point_on_triangle_covers_every_region() {
        let tri = [[0.0f32, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let cases: [([f64; 3], [f64; 3], [f64; 3]); 7] = [
            ([0.5, 0.5, 3.0], [0.5, 0.5, 0.0], [0.5, 0.25, 0.25]), // face
            ([-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), // vertex a
            ([3.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // vertex b
            ([-1.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]),  // vertex c
            ([1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]),  // edge ab
            ([-1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]),  // edge ac
            ([2.0, 2.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.5, 0.5]),   // edge bc
        ];
        for (p, expect_q, expect_bary) in cases {
            let (q, bary) = closest_point_on_triangle(tri, p);
            for k in 0..3 {
                assert!((q[k] - expect_q[k]).abs() < 1e-12, "{p:?}: point {q:?}");
                assert!(
                    (bary[k] - expect_bary[k]).abs() < 1e-12,
                    "{p:?}: bary {bary:?}"
                );
            }
        }
    }

    #[test]
    fn vertex_colours_interpolate_at_the_nearest_surface_point() {
        let mut mesh = cube(0.0, 1.0);
        // Colour = position: a probe off a face reads the face's lateral
        // coordinates, interpolated.
        mesh.vertex_fields.push(MeshField {
            name: COLOR_FIELD_NAME.to_string(),
            components: 3,
            data: mesh.positions.clone(),
        });
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.has_color());

        let close = |a: [f32; 3], b: [f64; 3]| (0..3).all(|k| (a[k] as f64 - b[k]).abs() < 1e-6);
        // Just outside the +x face at (y, z) = (0.3, 0.7): colour (1, 0.3, 0.7).
        let c = view.nearest_color([1.1, 0.3, 0.7]).unwrap();
        assert!(close(c, [1.0, 0.3, 0.7]), "{c:?}");
        // Just inside the same face.
        let c = view.nearest_color([0.95, 0.3, 0.7]).unwrap();
        assert!(close(c, [1.0, 0.3, 0.7]), "{c:?}");
        // Off the -z face.
        let c = view.nearest_color([0.2, 0.6, -0.4]).unwrap();
        assert!(close(c, [0.2, 0.6, 0.0]), "{c:?}");
        // Beyond a corner: the corner's colour.
        let c = view.nearest_color([3.0, 3.0, 3.0]).unwrap();
        assert!(close(c, [1.0, 1.0, 1.0]), "{c:?}");
        // A colour field with the wrong arity is ignored, not an error.
        let mut bad = cube(0.0, 1.0);
        bad.vertex_fields.push(MeshField {
            name: COLOR_FIELD_NAME.to_string(),
            components: 4,
            data: vec![0.5; 32],
        });
        assert!(!PayloadView::new(&plain(&bad)).unwrap().has_color());
    }

    #[test]
    fn nearest_colour_bvh_matches_brute_force_on_a_sphere() {
        let mut mesh = uv_sphere(16, 24);
        // Colour = unit position remapped to 0..1, so every triangle has a
        // distinct colour and any wrong triangle is visible.
        mesh.vertex_fields.push(MeshField {
            name: COLOR_FIELD_NAME.to_string(),
            components: 3,
            data: mesh.positions.iter().map(|v| (v + 1.0) * 0.5).collect(),
        });
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();

        let mut state = 0xabcdefu64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 3.0 - 1.5
        };
        for _ in 0..500 {
            let p = [rand(), rand(), rand()];
            let (t, bary) = brute_force_closest(&mesh, p);
            let idx = mesh.triangle(t);
            let expect: [f64; 3] = std::array::from_fn(|k| {
                (0..3)
                    .map(|c| bary[c] * ((mesh.position(idx[c] as usize)[k] + 1.0) * 0.5))
                    .sum()
            });
            let got = view.nearest_color(p).unwrap();
            for k in 0..3 {
                assert!(
                    (got[k] as f64 - expect[k]).abs() < 1e-5,
                    "nearest colour at {p:?}: {got:?} vs {expect:?}"
                );
            }
        }
    }

    /// A unit cube with four vertices per face, each face mapped to the
    /// whole texture: `u` along the lower lateral axis, `v` along the
    /// higher one.
    pub(crate) fn uv_cube() -> TriMesh {
        let mut positions = Vec::new();
        let mut uv = Vec::new();
        let mut indices = Vec::new();
        for axis in 0..3 {
            let (ua, va) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            // (u, v, axis) is right-handed for x and z, left-handed for y.
            let right_handed = axis != 1;
            for side in [0.0, 1.0] {
                let base = (positions.len() / 3) as u32;
                for (cu, cv) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                    let mut p = [0.0; 3];
                    p[axis] = side;
                    p[ua] = cu;
                    p[va] = cv;
                    positions.extend(p);
                    uv.extend([cu, cv]);
                }
                let outward = (side == 1.0) == right_handed;
                if outward {
                    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                } else {
                    indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
                }
            }
        }
        TriMesh {
            positions,
            indices,
            vertex_fields: vec![MeshField {
                name: UV_FIELD_NAME.to_string(),
                components: 2,
                data: uv,
            }],
            face_fields: vec![],
            textures: vec![MeshTexture {
                name: "checker.png".to_string(),
                media_type: "image/png".to_string(),
                bytes: vec![1], // never decoded here: tests inject the pixels
            }],
        }
    }

    /// Probe every face of [`uv_cube`] at its four quadrant centres and
    /// compare with the 2×2 `texels` (row-major, top row first).
    fn check_uv_cube(view: &PayloadView, texels: [[u8; 3]; 4], tol: f32) {
        for axis in 0..3 {
            let (ua, va) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            for (side, off) in [(0.0, -0.05), (1.0, 1.05)] {
                for (u, v) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let mut p = [0.0; 3];
                    p[axis] = off;
                    p[ua] = u;
                    p[va] = v;
                    let col = if u < 0.5 { 0 } else { 1 };
                    let row = if v < 0.5 { 1 } else { 0 }; // v up, rows down
                    let expect = texels[row * 2 + col];
                    let got = view.nearest_color(p).unwrap();
                    for k in 0..3 {
                        assert!(
                            (got[k] - expect[k] as f32 / 255.0).abs() <= tol,
                            "face axis {axis} side {side} uv ({u}, {v}): {got:?} vs {expect:?}"
                        );
                    }
                }
            }
        }
    }

    fn textured(mesh: &TriMesh, image: &TextureImage, format: TextureFormat) -> Vec<u8> {
        let decode = |_: &MeshTexture| Some(image.clone());
        build_payload(
            mesh,
            &BuildOptions {
                texture_format: format,
                texture_max_size: 0,
                decode_texture: Some(&decode),
            },
        )
        .unwrap()
    }

    #[test]
    fn textured_cube_reads_the_checker_rgb8() {
        let mesh = uv_cube();
        let texels = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let image = TextureImage {
            width: 2,
            height: 2,
            rgb: texels.concat(),
        };
        let payload = textured(&mesh, &image, TextureFormat::Rgb8);
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.has_color());
        check_uv_cube(&view, texels, 1e-6);

        // Repeat wrapping: a UV outside [0, 1] reads the same texel, and
        // bilinear filtering between two texel centres averages them.
        let color = view.color.unwrap();
        assert_eq!(
            view.sample_texture(&color, [1.25, -0.75]),
            view.sample_texture(&color, [0.25, 0.25])
        );
        let mid = view.sample_texture(&color, [0.5, 0.75]); // between red and green
        assert!(
            (mid[0] - 0.5).abs() < 1e-6 && (mid[1] - 0.5).abs() < 1e-6,
            "{mid:?}"
        );
    }

    #[test]
    fn textured_cube_reads_the_checker_bc1() {
        let mesh = uv_cube();
        let texels = [[255u8, 0, 0], [0, 0, 255], [0, 0, 255], [255, 0, 0]];
        let image = TextureImage {
            width: 2,
            height: 2,
            rgb: texels.concat(),
        };
        let payload = textured(&mesh, &image, TextureFormat::Bc1);
        let view = PayloadView::new(&payload).unwrap();
        check_uv_cube(&view, texels, 1e-6);
        // One 4x4 block for a 2x2 image.
        let rgb8 = textured(&mesh, &image, TextureFormat::Rgb8);
        assert_eq!(payload.len(), rgb8.len() - 12 + 8);
    }

    #[test]
    fn texture_preference_and_fallbacks() {
        let mut mesh = uv_cube();
        mesh.vertex_fields.push(MeshField {
            name: COLOR_FIELD_NAME.to_string(),
            components: 3,
            data: vec![0.25; 24 * 3],
        });
        let image = TextureImage {
            width: 1,
            height: 1,
            rgb: vec![255, 255, 255],
        };
        // Texture wins when a decoder is supplied...
        let payload = textured(&mesh, &image, TextureFormat::Rgb8);
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.nearest_color([0.5, 0.5, 1.5]), Some([1.0, 1.0, 1.0]));
        // ...the vertex colour otherwise...
        let payload = plain(&mesh);
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(
            view.nearest_color([0.5, 0.5, 1.5]),
            Some([0.25, 0.25, 0.25])
        );
        // ...and an undecodable texture falls back the same way (the
        // decoder owns the warning), down to no colour at all.
        let fail = |_: &MeshTexture| None;
        let payload = build_payload(
            &mesh,
            &BuildOptions {
                decode_texture: Some(&fail),
                ..BuildOptions::default()
            },
        )
        .unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(
            view.nearest_color([0.5, 0.5, 1.5]),
            Some([0.25, 0.25, 0.25])
        );
        let payload = build_payload(
            &uv_cube(),
            &BuildOptions {
                decode_texture: Some(&fail),
                ..BuildOptions::default()
            },
        )
        .unwrap();
        assert!(!PayloadView::new(&payload).unwrap().has_color());
    }

    #[test]
    fn texture_max_size_shrinks_the_atlas() {
        let mesh = uv_cube();
        let image = TextureImage {
            width: 8,
            height: 4,
            rgb: vec![200; 8 * 4 * 3],
        };
        let decode = |_: &MeshTexture| Some(image.clone());
        let payload = build_payload(
            &mesh,
            &BuildOptions {
                texture_format: TextureFormat::Rgb8,
                texture_max_size: 3,
                decode_texture: Some(&decode),
            },
        )
        .unwrap();
        let view = PayloadView::new(&payload).unwrap();
        let color = view.color.unwrap();
        assert_eq!((color.width, color.height), (2, 1));
        let c = view.nearest_color([0.5, 0.5, 1.5]).unwrap();
        assert!((c[0] - 200.0 / 255.0).abs() < 1e-6, "{c:?}");
    }

    /// Payload sizes for a synthetic 1024² texture in each format, printed
    /// for the record and pinned so the layout cannot silently change.
    #[test]
    fn payload_sizes_for_a_1024_square_texture() {
        let mesh = uv_cube();
        let image = TextureImage {
            width: 1024,
            height: 1024,
            rgb: vec![77; 1024 * 1024 * 3],
        };
        let geometry = plain(&cube(0.0, 1.0)).len();
        let header_and_uvs = COLOR_HEADER_LEN + 12 * 6 * 4;
        let rgb8 = textured(&mesh, &image, TextureFormat::Rgb8).len();
        let bc1 = textured(&mesh, &image, TextureFormat::Bc1).len();
        let uv_geometry = plain(&mesh).len();
        eprintln!(
            "payload sizes: geometry-only (12 tris) {geometry} B, uv cube {uv_geometry} B, Rgb8 {rgb8} B, Bc1 {bc1} B"
        );
        assert_eq!(rgb8, uv_geometry + header_and_uvs + 1024 * 1024 * 3);
        assert_eq!(bc1, uv_geometry + header_and_uvs + 256 * 256 * 8);
    }
}
