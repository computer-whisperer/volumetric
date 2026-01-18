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

    /// Get the current capacity in elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Ensure the buffer can hold at least `required` elements.
    /// Reallocates with 2x growth if needed.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.capacity {
            return;
        }

        // Calculate new capacity with 2x growth, minimum 64 elements
        let new_capacity = required.max(self.capacity * 2).max(64);
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

    /// Upload data to the GPU buffer. Reallocates if needed.
    ///
    /// Returns true if the buffer was reallocated.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) -> bool {
        if data.is_empty() {
            self.len = 0;
            return false;
        }

        let reallocated = data.len() > self.capacity;
        self.ensure_capacity(device, data.len());

        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        }

        self.len = data.len();
        reallocated
    }

    /// Upload data starting at a specific offset (in elements).
    /// Does not resize the buffer - caller must ensure capacity.
    pub fn upload_at(&mut self, queue: &wgpu::Queue, offset: usize, data: &[T]) {
        if data.is_empty() {
            return;
        }

        if let Some(buffer) = &self.buffer {
            let byte_offset = (offset * std::mem::size_of::<T>()) as u64;
            queue.write_buffer(buffer, byte_offset, bytemuck::cast_slice(data));
        }

        // Update len if we wrote past the current end
        self.len = self.len.max(offset + data.len());
    }

    /// Clear the buffer contents (sets len to 0 but keeps allocation).
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

/// A uniform buffer with change detection.
///
/// Only uploads to GPU when the data has changed.
pub struct UniformBuffer<T: Pod + PartialEq> {
    buffer: Option<wgpu::Buffer>,
    cached_value: Option<T>,
    label: &'static str,
}

impl<T: Pod + PartialEq> UniformBuffer<T> {
    /// Create a new uniform buffer with the given label.
    pub fn new(label: &'static str) -> Self {
        Self {
            buffer: None,
            cached_value: None,
            label,
        }
    }

    /// Get the underlying buffer, creating it if needed.
    pub fn get_or_create(&mut self, device: &wgpu::Device) -> &wgpu::Buffer {
        if self.buffer.is_none() {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: std::mem::size_of::<T>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.buffer.as_ref().unwrap()
    }

    /// Get the underlying buffer if it exists.
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Upload data if it has changed. Returns true if upload occurred.
    pub fn upload_if_changed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        value: &T,
    ) -> bool {
        // Check if value changed
        if let Some(cached) = &self.cached_value {
            if cached == value {
                return false;
            }
        }

        // Ensure buffer exists
        self.get_or_create(device);

        // Upload
        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(value));
        }

        self.cached_value = Some(*value);
        true
    }

    /// Force upload regardless of whether value changed.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, value: &T) {
        self.get_or_create(device);

        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(value));
        }

        self.cached_value = Some(*value);
    }

    /// Get the cached value, if any.
    pub fn cached_value(&self) -> Option<&T> {
        self.cached_value.as_ref()
    }
}

/// A static buffer that is created once with initial data.
pub struct StaticBuffer<T: Pod> {
    buffer: wgpu::Buffer,
    len: usize,
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
            len: data.len(),
            _marker: PhantomData,
        }
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
