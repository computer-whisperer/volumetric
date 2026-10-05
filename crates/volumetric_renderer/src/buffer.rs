//! Dynamic GPU buffer management.
//!
//! Provides a resizable buffer that grows as needed to accommodate data uploads.

use bytemuck::{Pod, Zeroable};
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

/// A dynamically-sized GPU buffer that grows as needed.
///
/// Uses a 2x growth strategy to minimize reallocations while keeping
/// memory usage reasonable.
pub struct DynamicBuffer<T: Pod> {
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    len: usize,
    usage: wgpu::BufferUsages,
    label: &'static str,
    _marker: PhantomData<T>,
}

impl<T: Pod> DynamicBuffer<T> {
    /// Create a new dynamic buffer with the given usage flags and label.
    pub fn new(usage: wgpu::BufferUsages, label: &'static str) -> Self {
        Self {
            buffer: None,
            capacity: 0,
            len: 0,
            usage,
            label,
            _marker: PhantomData,
        }
    }

    /// Get the underlying buffer, if allocated.
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Get the current number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The most elements a buffer of `T` may hold under the device's
    /// `max_buffer_size` limit.
    pub fn max_elements(device: &wgpu::Device) -> usize {
        (device.limits().max_buffer_size / std::mem::size_of::<T>().max(1) as u64) as usize
    }

    /// Ensure the buffer can hold at least `required` elements.
    /// Reallocates with 2x growth if needed. Both the growth headroom and
    /// the required size are clamped to the device's `max_buffer_size`
    /// limit — allocating past it is a wgpu validation panic.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.capacity {
            return;
        }

        // Calculate new capacity with 2x growth, minimum 64 elements,
        // clamped to the device limit.
        let new_capacity = required
            .max(self.capacity * 2)
            .max(64)
            .min(Self::max_elements(device));
        if new_capacity <= self.capacity {
            return; // Already at the device limit.
        }
        let byte_size = new_capacity * std::mem::size_of::<T>();

        // Create new buffer
        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: byte_size as u64,
            usage: self.usage,
            mapped_at_creation: false,
        });

        self.buffer = Some(new_buffer);
        self.capacity = new_capacity;
    }

    /// Upload data to the GPU buffer. Reallocates if needed. Data beyond
    /// the device's `max_buffer_size` limit is dropped rather than
    /// panicking wgpu.
    ///
    /// Returns the number of elements actually uploaded.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) -> usize {
        if data.is_empty() {
            self.len = 0;
            return 0;
        }

        self.ensure_capacity(device, data.len());
        let count = data.len().min(self.capacity);

        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&data[..count]));
        }

        self.len = count;
        count
    }
}

/// A static buffer that is created once with initial data.
pub struct StaticBuffer<T: Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: Pod> StaticBuffer<T> {
    /// Create a new static buffer with the given data.
    pub fn new(
        device: &wgpu::Device,
        data: &[T],
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });

        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Quad vertices for instanced rendering (used by both lines and points).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct QuadVertex {
    /// x: 0=start/center, 1=end (for lines, position along line)
    /// y: -1=left/bottom, +1=right/top (perpendicular offset)
    pub corner: [f32; 2],
    /// UV coordinates for texturing
    pub uv: [f32; 2],
}

/// Create quad vertices for instanced rendering.
pub const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex {
        corner: [0.0, -1.0],
        uv: [0.0, 0.0],
    }, // start/center, left/bottom
    QuadVertex {
        corner: [0.0, 1.0],
        uv: [0.0, 1.0],
    }, // start/center, right/top
    QuadVertex {
        corner: [1.0, -1.0],
        uv: [1.0, 0.0],
    }, // end, left/bottom
    QuadVertex {
        corner: [1.0, 1.0],
        uv: [1.0, 1.0],
    }, // end, right/top
];

/// Quad indices for two triangles.
pub const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];
