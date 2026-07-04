use crate::metal_atlas::MetalAtlas;
use anyhow::Result;
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
use gpui::{
    AtlasTextureId, BackdropBlur, Background, Bounds, ContentMask, DevicePixels, MonochromeSprite,
    PaintSurface, Path, Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledPixels, Scene,
    ShaderPass, Shadow, Size, Surface, Underline, point, size,
};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;

use core_foundation::{
    base::{CFType, TCFType},
    boolean::CFBoolean,
    dictionary::CFDictionary,
    number::CFNumber,
    string::CFString,
};
use core_video::{
    metal_texture::{CVMetalTextureGetTexture, kCVMetalTextureUsage},
    metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::{
        CVPixelBuffer, kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
        kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    },
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, MTLTextureUsage,
    NSRange, RenderPassColorAttachmentDescriptorRef,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use std::{cell::Cell, cell::RefCell, collections::HashMap, ffi::c_void, mem, ptr, sync::Arc};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

pub(crate) type Context = Arc<Mutex<InstanceBufferPool>>;
pub(crate) type Renderer = MetalRenderer;

pub(crate) unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    /// Pipeline for sampling a single-plane BGRA8 surface (the zero-copy
    /// live-thumbnail path), as opposed to the YUV biplanar `surfaces` pipeline.
    surfaces_pipeline_state_bgra: metal::RenderPipelineState,
    /// Composites the captured-and-blurred backdrop back within a rounded rect (frosted-glass
    /// overlays). Samples `backdrop_blur_texture`; same premultiplied blend as path sprites.
    backdrop_blur_pipeline_state: metal::RenderPipelineState,
    /// Captures the drawable into the scene texture vertically flipped, before a scene-sampling shader
    /// pass reads it as `iChannel0` (the shader lib's Shadertoy y-flip expects a bottom-left origin, and
    /// a Metal blit can't invert). No blending — it overwrites the whole capture target.
    scene_flip_pipeline_state: metal::RenderPipelineState,
    /// The static fullscreen-quad vertex function paired with every runtime-compiled shader-pass
    /// fragment (each shader-pass pipeline mixes this vertex with the shader's own fragment). Located
    /// once at startup — Metal allows a pipeline's vertex + fragment to come from different libraries.
    shader_pass_vertex_function: metal::Function,
    /// Linear / clamp-to-edge sampler bound as `iChannelN` for scene-sampling shader passes.
    shader_pass_sampler: metal::SamplerState,
    /// Compiled-pipeline cache for shader passes, keyed by [`gpui::ShaderPass::shader_id`]. Each distinct
    /// shader is compiled from MSL (`new_library_with_source`) into a pipeline exactly once and reused.
    shader_pass_pipelines: RefCell<HashMap<u64, metal::RenderPipelineState>>,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    /// Off-screen, viewport-sized copy of the drawable captured just before compositing a
    /// backdrop-blur primitive, so the composite pass can sample the backdrop (the drawable itself
    /// is the render target and cannot be sampled in the same pass). Reused/resized per frame.
    backdrop_blur_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow the drawable to be sampled / blit-copied: visual tests read pixels back without
        // ScreenCaptureKit, and backdrop-blur captures the rendered backdrop into an off-screen
        // texture before compositing. (A framebuffer-only drawable can be neither sampled nor copied.)
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state_bgra = build_pipeline_state(
            &device,
            &library,
            "surfaces_bgra",
            "surface_vertex",
            "surface_fragment_bgra",
            MTLPixelFormat::BGRA8Unorm,
        );
        // Premultiplied-alpha blend (source One, dest OneMinusSourceAlpha) — the same as path
        // sprites, since the composite returns premultiplied `backdrop * mask`.
        let backdrop_blur_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "backdrop_blur",
            "backdrop_blur_vertex",
            "backdrop_blur_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        // No blending: the flip pass overwrites the whole scene-capture texture with the drawable.
        let scene_flip_pipeline_state = build_copy_pipeline_state(
            &device,
            &library,
            "scene_flip",
            "scene_flip_vertex",
            "scene_flip_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        // The static fullscreen-quad vertex shader-pass pipelines pair with each runtime-compiled MSL
        // fragment. Located once; the per-shader fragment is compiled + cached lazily in `shader_pass_pipeline`.
        let shader_pass_vertex_function = library
            .get_function("shader_pass_vertex", None)
            .expect("error locating shader_pass_vertex");
        let shader_pass_sampler = {
            let descriptor = metal::SamplerDescriptor::new();
            descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
            descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
            descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
            descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
            device.new_sampler(&descriptor)
        };

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            surfaces_pipeline_state_bgra,
            backdrop_blur_pipeline_state,
            scene_flip_pipeline_state,
            shader_pass_vertex_function,
            shader_pass_sampler,
            shader_pass_pipelines: RefCell::new(HashMap::new()),
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            backdrop_blur_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            let ns_size = NSSize {
                width: size.width.0 as f64,
                height: size.height.0 as f64,
            };
            unsafe {
                let _: () = msg_send![
                    layer.as_ref(),
                    setDrawableSize: ns_size
                ];
            }
        }
        self.update_path_intermediate_textures(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            self.backdrop_blur_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));
        // Same descriptor (RenderTarget | ShaderRead, BGRA8Unorm): the off-screen target the
        // drawable is captured into before compositing a backdrop blur.
        self.backdrop_blur_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    if self.presents_with_transaction {
                        command_buffer.commit();
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    return;
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    // Commit and wait for completion without presenting
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    // Read pixels from the texture
                    let texture = drawable.texture();
                    let width = texture.width() as u32;
                    let height = texture.height() as u32;
                    let bytes_per_row = width as usize * 4;
                    let buffer_size = height as usize * bytes_per_row;

                    let mut pixels = vec![0u8; buffer_size];

                    let region = metal::MTLRegion {
                        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        size: metal::MTLSize {
                            width: width as u64,
                            height: height as u64,
                            depth: 1,
                        },
                    };

                    texture.get_bytes(
                        pixels.as_mut_ptr() as *mut std::ffi::c_void,
                        bytes_per_row as u64,
                        region,
                        0,
                    );

                    // Convert BGRA to RGBA (swap B and R channels)
                    for chunk in pixels.chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }

                    return RgbaImage::from_raw(width, height, pixels).ok_or_else(|| {
                        anyhow::anyhow!("Failed to create RgbaImage from pixel data")
                    });
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        anyhow::bail!("instance buffer size grew too large: {}", buffer_size);
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    /// Renders a scene into a fresh IOSurface-backed BGRA8 [`CVPixelBuffer`] and
    /// returns it without reading the pixels back to the CPU.
    ///
    /// The rendered frame stays in GPU/IOSurface memory, so it can be sampled
    /// directly by a later [`PaintSurface`] primitive through the BGRA surface
    /// pipeline — the zero-copy path that draws a live, uniformly scaled-down
    /// thumbnail of a session's full-size view. Unlike [`render_scene_to_image`],
    /// this is a production capability (not test-gated): the live GPU client
    /// produces a session thumbnail this way each frame the panel changes.
    ///
    /// [`render_scene_to_image`]: Self::render_scene_to_image
    pub fn render_scene_to_surface(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<CVPixelBuffer> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_surface: {:?}", size);
        }
        let width = size.width.0 as usize;
        let height = size.height.0 as usize;

        // Path primitives render through intermediate textures sized to the target.
        self.update_path_intermediate_textures(size);

        // 1. Allocate an IOSurface-backed, Metal-compatible BGRA pixel buffer.
        let pixel_buffer =
            create_metal_surface_pixel_buffer(kCVPixelFormatType_32BGRA, width, height)?;

        // 2. Wrap it as a Metal render-target texture via the CoreVideo texture
        //    cache. The `kCVMetalTextureUsage` attribute is what grants
        //    RenderTarget usage — without it the cache returns a sample-only
        //    texture that cannot be attached as a color attachment.
        let texture_attributes = bgra_render_target_attributes();
        let cv_texture = self
            .core_video_texture_cache
            .create_texture_from_image(
                pixel_buffer.as_concrete_TypeRef(),
                Some(&texture_attributes),
                MTLPixelFormat::BGRA8Unorm,
                width,
                height,
                0,
            )
            .map_err(|cv_return| {
                anyhow::anyhow!("CVMetalTextureCache create failed: CVReturn({cv_return})")
            })?;
        // SAFETY: `cv_texture` owns the CVMetalTexture; the borrowed MTLTexture is
        // valid while `cv_texture` is alive. It is held until the end of this
        // function, after the GPU work below completes.
        let target_texture = unsafe {
            let ptr = CVMetalTextureGetTexture(cv_texture.as_concrete_TypeRef());
            anyhow::ensure!(!ptr.is_null(), "CVMetalTextureGetTexture returned null");
            metal::TextureRef::from_ptr(ptr as *mut _)
        };

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives_to_texture(scene, &mut instance_buffer, target_texture, size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    // Wait so the surface is fully rendered before a consumer
                    // samples it. The pixels live in the IOSurface — no read-back.
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    // `cv_texture` (kept alive above) is dropped here, after the
                    // GPU finished writing the surface.
                    return Ok(pixel_buffer);
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        anyhow::bail!("instance buffer size grew too large: {}", buffer_size);
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    /// Renders `scene` into a fresh BGRA surface sized to this renderer's layer —
    /// the window-backed companion to `render_scene_to_surface`. Mirrors the
    /// drawable-size derivation of `render_to_image` but keeps the result on the
    /// GPU as a reusable `CVPixelBuffer` instead of reading pixels back.
    pub fn render_to_surface(&mut self, scene: &Scene) -> Result<CVPixelBuffer> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_surface requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        self.render_scene_to_surface(scene, viewport_size)
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Update path intermediate textures for this size
        self.update_path_intermediate_textures(size);

        // Create an offscreen texture as render target
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
        let target_texture = self.device.new_texture(&texture_descriptor);

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives_to_texture(scene, &mut instance_buffer, &target_texture, size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    // On discrete GPUs (non-unified memory), Managed textures
                    // require an explicit blit synchronize before the CPU can
                    // read back the rendered data. Without this, get_bytes
                    // returns stale zeros.
                    if !self.is_unified_memory {
                        let blit = command_buffer.new_blit_command_encoder();
                        blit.synchronize_resource(&target_texture);
                        blit.end_encoding();
                    }

                    // Commit and wait for completion
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    // Read pixels from the texture
                    let width = size.width.0 as u32;
                    let height = size.height.0 as u32;
                    let bytes_per_row = width as usize * 4;
                    let buffer_size = height as usize * bytes_per_row;

                    let mut pixels = vec![0u8; buffer_size];

                    let region = metal::MTLRegion {
                        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        size: metal::MTLSize {
                            width: width as u64,
                            height: height as u64,
                            depth: 1,
                        },
                    };

                    target_texture.get_bytes(
                        pixels.as_mut_ptr() as *mut std::ffi::c_void,
                        bytes_per_row as u64,
                        region,
                        0,
                    );

                    // Convert BGRA to RGBA (swap B and R channels)
                    for chunk in pixels.chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }

                    return RgbaImage::from_raw(width, height, pixels).ok_or_else(|| {
                        anyhow::anyhow!("Failed to create RgbaImage from pixel data")
                    });
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        anyhow::bail!("instance buffer size grew too large: {}", buffer_size);
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        self.update_path_intermediate_textures(size);

        let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
            texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
        });
        if needs_new_target {
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
        }
        let target_texture = self
            .headless_render_target
            .clone()
            .expect("just ensured the render target exists");

        loop {
            let mut instance_buffer = self
                .instance_buffer_pool
                .lock()
                .acquire(&self.device, self.is_unified_memory);

            let command_buffer =
                self.draw_primitives_to_texture(scene, &mut instance_buffer, &target_texture, size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    // Commit without waiting, mirroring presentation to a real
                    // window where the CPU doesn't block on the GPU.
                    command_buffer.commit();
                    return Ok(());
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        anyhow::bail!("instance buffer size grew too large: {}", buffer_size);
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        self.draw_primitives_to_texture(scene, instance_buffer, drawable.texture(), viewport_size)
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };
        let mut instance_offset = 0;

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., alpha));
            },
        );

        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(
                    &scene.shadows[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::BackdropBlurs(range) => {
                    let blurs = &scene.backdrop_blurs[range];
                    // The drawable now holds everything below the overlay. End the pass and copy it
                    // into an off-screen texture we can sample — the drawable is this pass's render
                    // target, so it cannot also be a shader input.
                    command_encoder.end_encoding();

                    let captured = if let Some(ref backdrop_texture) = self.backdrop_blur_texture {
                        let blit = command_buffer.new_blit_command_encoder();
                        blit.copy_from_texture(
                            texture,
                            0,
                            0,
                            metal::MTLOrigin { x: 0, y: 0, z: 0 },
                            metal::MTLSize {
                                width: viewport_size.width.0 as u64,
                                height: viewport_size.height.0 as u64,
                                depth: 1,
                            },
                            backdrop_texture,
                            0,
                            0,
                            metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        );
                        blit.end_encoding();
                        true
                    } else {
                        false
                    };

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );

                    if captured {
                        self.draw_backdrop_blurs(
                            blurs,
                            instance_buffer,
                            &mut instance_offset,
                            viewport_size,
                            command_encoder,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Quads(range) => self.draw_quads(
                    &scene.quads[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_buffer,
                    );

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );

                    if did_draw {
                        self.draw_paths_from_intermediate(
                            paths,
                            instance_buffer,
                            &mut instance_offset,
                            viewport_size,
                            command_encoder,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(
                    &scene.underlines[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        &scene.monochrome_sprites[range],
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        &scene.polychrome_sprites[range],
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range],
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::ShaderPasses(range) => {
                    let passes = &scene.shader_passes[range];
                    if passes.iter().any(|pass| pass.samples_scene) {
                        // A scene-sampling pass reads the drawable-so-far as iChannel0; the drawable is
                        // this pass's own render target and cannot also be a shader input, so capture it
                        // into an off-screen texture first (the same dance as backdrop-blur). Unlike
                        // backdrop-blur, capture it VERTICALLY FLIPPED: the shader lib y-flips fragCoord
                        // to Shadertoy's bottom-left origin, so a straight top-left copy would sample the
                        // scene upside down. A Metal blit can't invert, so render a full-screen flip.
                        command_encoder.end_encoding();
                        if let Some(ref scene_texture) = self.backdrop_blur_texture {
                            let flip_encoder = new_command_encoder_for_texture(
                                command_buffer,
                                scene_texture,
                                viewport_size,
                                |color_attachment| {
                                    color_attachment
                                        .set_load_action(metal::MTLLoadAction::DontCare);
                                },
                            );
                            flip_encoder.set_render_pipeline_state(&self.scene_flip_pipeline_state);
                            flip_encoder.set_vertex_buffer(0, Some(&self.unit_vertices), 0);
                            flip_encoder.set_fragment_texture(0, Some(texture));
                            flip_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
                            flip_encoder.end_encoding();
                        }
                        command_encoder = new_command_encoder_for_texture(
                            command_buffer,
                            texture,
                            viewport_size,
                            |color_attachment| {
                                color_attachment.set_load_action(metal::MTLLoadAction::Load);
                            },
                        );
                    }
                    self.draw_shader_passes(passes, viewport_size, command_encoder)
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            };
            if !ok {
                command_encoder.end_encoding();
                anyhow::bail!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.end_encoding();

        if !self.is_unified_memory {
            // Sync the instance buffer to the GPU
            instance_buffer.metal_buffer.did_modify_range(NSRange {
                location: 0,
                length: instance_offset as NSUInteger,
            });
        }

        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);

        align_offset(instance_offset);
        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertices_bytes_len = mem::size_of_val(vertices.as_slice());
        let next_offset = *instance_offset + vertices_bytes_len;
        if next_offset > instance_buffer.size {
            command_encoder.end_encoding();
            return false;
        }
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                buffer_contents,
                vertices_bytes_len,
            );
        }
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );
        *instance_offset = next_offset;

        command_encoder.end_encoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let shadow_bytes_len = mem::size_of_val(shadows);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    /// Composite the captured (and, in later passes, blurred) backdrop back within each blur's
    /// rounded rect. Samples `backdrop_blur_texture` at screen-uv (passed through from the vertex
    /// stage) and masks to the rounded bounds via the shared `quad_sdf`. Run inside a `Load` pass
    /// on the drawable, after the backdrop has been captured.
    fn draw_backdrop_blurs(
        &self,
        blurs: &[BackdropBlur],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if blurs.is_empty() {
            return true;
        }
        let Some(ref backdrop_texture) = self.backdrop_blur_texture else {
            return false;
        };
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.backdrop_blur_pipeline_state);
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            BackdropBlurInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_texture(
            BackdropBlurInputIndex::BackdropTexture as u64,
            Some(backdrop_texture),
        );

        let blur_bytes_len = mem::size_of_val(blurs);
        let next_offset = *instance_offset + blur_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(blurs.as_ptr() as *const u8, buffer_contents, blur_bytes_len);
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            blurs.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    /// Get, or lazily compile + cache, the render pipeline for a shader pass. On first sight of a
    /// `shader_id` the MSL is compiled (`new_library_with_source`) and its fragment entry is paired with
    /// the static `shader_pass_vertex` into a pipeline. A shader that fails to compile is logged and
    /// skipped (returns `None`) — a bad shader never crashes the renderer.
    fn shader_pass_pipeline(&self, pass: &ShaderPass) -> Option<metal::RenderPipelineState> {
        if let Some(pipeline) = self.shader_pass_pipelines.borrow().get(&pass.shader_id) {
            return Some(pipeline.clone());
        }
        let library = match self
            .device
            .new_library_with_source(&pass.msl, &metal::CompileOptions::new())
        {
            Ok(library) => library,
            Err(err) => {
                log::error!("shader pass {} failed to compile: {err}", pass.shader_id);
                return None;
            }
        };
        let fragment_function = match library.get_function(&pass.fragment_entry, None) {
            Ok(function) => function,
            Err(err) => {
                log::error!(
                    "shader pass {} missing fragment entry `{}`: {err}",
                    pass.shader_id,
                    pass.fragment_entry
                );
                return None;
            }
        };

        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_label("shader_pass");
        descriptor.set_vertex_function(Some(self.shader_pass_vertex_function.as_ref()));
        descriptor.set_fragment_function(Some(fragment_function.as_ref()));
        let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // No blending: a Shadertoy-convention shader outputs the final colour — a background fills the
        // target, a post-process samples the scene as iChannel0 and composites itself — so its output
        // replaces the target rather than alpha-blending over it.
        color_attachment.set_blending_enabled(false);

        let pipeline = match self.device.new_render_pipeline_state(&descriptor) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                log::error!(
                    "shader pass {} pipeline creation failed: {err}",
                    pass.shader_id
                );
                return None;
            }
        };
        self.shader_pass_pipelines
            .borrow_mut()
            .insert(pass.shader_id, pipeline.clone());
        Some(pipeline)
    }

    /// Draw each shader pass as its own fullscreen/bounded quad: bind its cached pipeline, the unit quad
    /// + the pass bounds/viewport (vertex), the packed uniform bytes (fragment `buffer(0)`), and — for a
    /// scene-sampling pass — the captured scene as `iChannel0` (`texture(0)`/`sampler(0)`). Distinct
    /// shaders have distinct pipelines, so passes are drawn individually, not instanced.
    fn draw_shader_passes(
        &self,
        passes: &[ShaderPass],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        for pass in passes {
            let Some(pipeline) = self.shader_pass_pipeline(pass) else {
                continue;
            };
            command_encoder.set_render_pipeline_state(&pipeline);
            command_encoder.set_vertex_buffer(
                ShaderPassInputIndex::Vertices as u64,
                Some(&self.unit_vertices),
                0,
            );
            command_encoder.set_vertex_bytes(
                ShaderPassInputIndex::Bounds as u64,
                mem::size_of::<Bounds<ScaledPixels>>() as u64,
                &pass.bounds as *const Bounds<ScaledPixels> as *const _,
            );
            command_encoder.set_vertex_bytes(
                ShaderPassInputIndex::ViewportSize as u64,
                mem::size_of_val(&viewport_size) as u64,
                &viewport_size as *const Size<DevicePixels> as *const _,
            );
            // The naga-emitted MSL binds its uniform block at fragment `buffer(0)` (the Wingman shader
            // lib's METAL_UNIFORM_BUFFER_SLOT); the fork mirrors that fixed slot.
            command_encoder.set_fragment_bytes(
                SHADER_PASS_UNIFORM_BUFFER_INDEX,
                pass.uniforms.len() as u64,
                pass.uniforms.as_ptr() as *const _,
            );
            if pass.samples_scene {
                if let Some(ref scene_texture) = self.backdrop_blur_texture {
                    command_encoder.set_fragment_texture(0, Some(scene_texture));
                    command_encoder.set_fragment_sampler_state(0, Some(&self.shader_pass_sampler));
                }
            }
            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
        }
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let quad_bytes_len = mem::size_of_val(quads);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        let Some(first_path) = paths.first() else {
            return true;
        };

        let Some(ref intermediate_texture) = self.path_intermediate_texture else {
            return false;
        };

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of_val(sprites.as_slice());
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;

        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let underline_bytes_len = mem::size_of_val(underlines);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        // Both surface pipelines share `surface_vertex`, so the vertex-stage
        // bindings are set once here; the per-surface pixel format then selects
        // the fragment pipeline (single-plane BGRA vs. YUV biplanar) in the loop.
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for surface in surfaces {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );
            let pixel_format = surface.image_buffer.get_pixel_format();

            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<Surface>();
            if next_offset > instance_buffer.size {
                return false;
            }

            command_encoder.set_vertex_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );

            // These CVMetalTextures must outlive the `draw_primitives` call below
            // (the command buffer retains the underlying MTLTextures at encode
            // time), so they are bound in the loop-body scope, not the branch.
            let bgra_texture;
            let y_texture;
            let cb_cr_texture;
            if pixel_format == kCVPixelFormatType_32BGRA {
                // Single-plane BGRA8 surface (e.g. produced by
                // `render_scene_to_surface`): sample it directly.
                command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state_bgra);
                bgra_texture = self
                    .core_video_texture_cache
                    .create_texture_from_image(
                        surface.image_buffer.as_concrete_TypeRef(),
                        None,
                        MTLPixelFormat::BGRA8Unorm,
                        surface.image_buffer.get_width(),
                        surface.image_buffer.get_height(),
                        0,
                    )
                    .unwrap();
                command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                    let texture = CVMetalTextureGetTexture(bgra_texture.as_concrete_TypeRef());
                    Some(metal::TextureRef::from_ptr(texture as *mut _))
                });
            } else {
                // YUV 4:2:0 biplanar surface (e.g. a decoded video frame):
                // convert to RGB in the fragment shader from its two planes.
                assert_eq!(pixel_format, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
                command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
                y_texture = self
                    .core_video_texture_cache
                    .create_texture_from_image(
                        surface.image_buffer.as_concrete_TypeRef(),
                        None,
                        MTLPixelFormat::R8Unorm,
                        surface.image_buffer.get_width_of_plane(0),
                        surface.image_buffer.get_height_of_plane(0),
                        0,
                    )
                    .unwrap();
                cb_cr_texture = self
                    .core_video_texture_cache
                    .create_texture_from_image(
                        surface.image_buffer.as_concrete_TypeRef(),
                        None,
                        MTLPixelFormat::RG8Unorm,
                        surface.image_buffer.get_width_of_plane(1),
                        surface.image_buffer.get_height_of_plane(1),
                        1,
                    )
                    .unwrap();
                command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                    let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                    Some(metal::TextureRef::from_ptr(texture as *mut _))
                });
                command_encoder.set_fragment_texture(
                    SurfaceInputIndex::CbCrTexture as u64,
                    unsafe {
                        let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                        Some(metal::TextureRef::from_ptr(texture as *mut _))
                    },
                );
            }

            unsafe {
                let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset)
                    as *mut SurfaceBounds;
                ptr::write(
                    buffer_contents,
                    SurfaceBounds {
                        bounds: surface.bounds,
                        content_mask: surface.content_mask,
                    },
                );
            }

            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&RenderPassColorAttachmentDescriptorRef),
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    configure_color_attachment(color_attachment);

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

/// Builds an IOSurface-backed, Metal-compatible `CVPixelBuffer` of the given
/// pixel format — for a BGRA8 offscreen render target a `PaintSurface` can later
/// sample (the live-thumbnail path), and reused by tests for other formats.
fn create_metal_surface_pixel_buffer(
    pixel_format: u32,
    width: usize,
    height: usize,
) -> Result<CVPixelBuffer> {
    let metal_key: CFString =
        unsafe { CFString::wrap_under_get_rule(kCVPixelBufferMetalCompatibilityKey) };
    let iosurface_key: CFString =
        unsafe { CFString::wrap_under_get_rule(kCVPixelBufferIOSurfacePropertiesKey) };
    // An empty IOSurface-properties dictionary requests default IOSurface backing.
    let iosurface_properties: CFDictionary<CFString, CFType> =
        CFDictionary::from_CFType_pairs(&[] as &[(CFString, CFType)]);
    let options = CFDictionary::from_CFType_pairs(&[
        (metal_key, CFBoolean::true_value().into_CFType()),
        (iosurface_key, iosurface_properties.into_CFType()),
    ]);
    CVPixelBuffer::new(pixel_format, width, height, Some(&options))
        .map_err(|cv_return| anyhow::anyhow!("CVPixelBufferCreate failed: CVReturn({cv_return})"))
}

/// Texture attributes that ask the CoreVideo texture cache for a
/// render-target-capable Metal texture. The `kCVMetalTextureUsage` value carries
/// the `MTLTextureUsage` bits; without `RenderTarget` the cache returns a
/// sample-only texture that cannot be attached as a color attachment.
fn bgra_render_target_attributes() -> CFDictionary<CFString, CFType> {
    let usage_key: CFString = unsafe { CFString::wrap_under_get_rule(kCVMetalTextureUsage) };
    let usage_bits = (MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead).bits() as i64;
    let usage_value = CFNumber::from(usage_bits);
    CFDictionary::from_CFType_pairs(&[(usage_key, usage_value.into_CFType())])
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// A render pipeline with blending DISABLED — the fragment's returned colour is written straight to
/// the target. For full-screen copies (e.g. the scene-flip capture) that overwrite every pixel, so
/// the drawable's own alpha never blends the copy against the target's prior contents.
fn build_copy_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum BackdropBlurInputIndex {
    Vertices = 0,
    BackdropBlurs = 1,
    ViewportSize = 2,
    BackdropTexture = 3,
}

/// The **vertex-stage** buffer slots for `shader_pass_vertex` (see shaders.metal). The fragment stage's
/// own slots (uniform block, iChannelN texture/sampler) are baked into the runtime-compiled MSL by naga
/// and are independent of these — see [`SHADER_PASS_UNIFORM_BUFFER_INDEX`].
#[repr(C)]
enum ShaderPassInputIndex {
    Vertices = 0,
    Bounds = 1,
    ViewportSize = 2,
}

/// The **fragment-stage** buffer slot the shader-pass uniform block binds to. Fixed at 0 to match the
/// Wingman `shader` lib's `METAL_UNIFORM_BUFFER_SLOT` (the naga MSL bakes `[[buffer(0)]]` into the block).
const SHADER_PASS_UNIFORM_BUFFER_INDEX: u64 = 0;

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod surface_roundtrip_tests {
    use super::*;
    use gpui::{BorderStyle, Corners, Edges, Hsla, bounds, solid_background};

    fn px_bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        bounds(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    }

    fn solid_quad(b: Bounds<ScaledPixels>, color: Hsla) -> Quad {
        Quad {
            order: 0,
            border_style: BorderStyle::default(),
            bounds: b,
            content_mask: ContentMask { bounds: b },
            background: solid_background(color),
            border_color: Hsla::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    /// Milestone 1 proof for the live-thumbnail path: render a scene into a BGRA
    /// IOSurface, then paint that surface — scaled down — through the new BGRA
    /// surface pipeline, and read the composite back.
    ///
    /// The source frame is left-red / right-blue. If the box shows red on its
    /// left and blue on its right with black around it, the whole frame was
    /// sampled as a *uniformly scaled-down miniature* (Exposé-style), positioned
    /// correctly — not re-laid-out at tile size. That is the property the tiles
    /// need, proven end to end with zero CPU pixel copies between the two passes.
    #[test]
    fn renders_scene_to_bgra_surface_then_paints_it_scaled_down() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);

        let frame = size(DevicePixels::from(256), DevicePixels::from(256));
        let red = Hsla {
            h: 0.0,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };
        let blue = Hsla {
            h: 240.0 / 360.0,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };

        // Scene 1: a full 256x256 frame — left half red, right half blue.
        let mut source = Scene::default();
        source.insert_primitive(solid_quad(px_bounds(0.0, 0.0, 128.0, 256.0), red));
        source.insert_primitive(solid_quad(px_bounds(128.0, 0.0, 128.0, 256.0), blue));
        source.finish();

        let surface = renderer
            .render_scene_to_surface(&source, frame)
            .expect("render_scene_to_surface should produce a BGRA surface");
        assert_eq!(surface.get_pixel_format(), kCVPixelFormatType_32BGRA);
        assert_eq!(surface.get_width(), 256);
        assert_eq!(surface.get_height(), 256);

        // Scene 2: paint that surface, scaled by 0.5, into a 128x128 box at (64,64).
        let box_bounds = px_bounds(64.0, 64.0, 128.0, 128.0);
        let mut composed = Scene::default();
        composed.insert_primitive(PaintSurface {
            order: 0,
            bounds: box_bounds,
            content_mask: ContentMask { bounds: box_bounds },
            image_buffer: surface,
        });
        composed.finish();

        let rendered = renderer
            .render_scene_to_image(&composed, frame)
            .expect("render_scene_to_image should composite the painted surface");

        let is_red = |p: &image::Rgba<u8>| p.0[0] > 180 && p.0[1] < 80 && p.0[2] < 80;
        let is_blue = |p: &image::Rgba<u8>| p.0[2] > 180 && p.0[0] < 80 && p.0[1] < 80;
        let is_black = |p: &image::Rgba<u8>| p.0[0] < 40 && p.0[1] < 40 && p.0[2] < 40;

        // Box spans x∈[64,192]; its left half samples the frame's red half, its
        // right half the blue half — the miniature preserves left/right layout.
        let box_left = rendered.get_pixel(80, 128);
        let box_right = rendered.get_pixel(176, 128);
        assert!(
            is_red(box_left),
            "box-left should sample the frame's red half, got {box_left:?}"
        );
        assert!(
            is_blue(box_right),
            "box-right should sample the frame's blue half, got {box_right:?}"
        );

        // Outside the box stays black: the thumbnail is placed, not full-bleed.
        let outside_tl = rendered.get_pixel(20, 20);
        let outside_br = rendered.get_pixel(236, 236);
        assert!(
            is_black(outside_tl),
            "top-left outside the box should be black, got {outside_tl:?}"
        );
        assert!(
            is_black(outside_br),
            "bottom-right outside the box should be black, got {outside_br:?}"
        );
    }

    /// Regression test for the scene-sampling shader-pass vertical flip. A `samples_scene` pass reads
    /// the drawable as `iChannel0`; the Wingman shader lib y-flips fragCoord to Shadertoy's bottom-left
    /// origin, so the renderer must capture the drawable vertically flipped or the sampled scene renders
    /// upside down. The pass here inverts the scene it samples: a red (top) / blue (bottom) frame must
    /// come back cyan (top) / yellow (bottom). A straight (un-flipped) capture would swap them; a pass
    /// that failed to compile and was skipped would leave the raw red/blue through — the colours catch both.
    #[test]
    fn scene_sampling_shader_pass_is_not_vertically_flipped() {
        // A contract-compliant scene-sampling fragment (mirrors the shader lib's emitted `main_` for
        // `fragColor = 1.0 - texture(iChannel0, uv)`): reads gl_FragCoord, applies the Shadertoy
        // bottom-left y-flip, samples iChannel0 at slot 0, inverts. Hand-written so the fork test needs
        // no naga dependency; the y-flip + binding slots are exactly what the capture path relies on.
        const INVERT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Globals { packed_float3 iResolution; };
struct Out { float4 color [[color(0)]]; };
fragment Out main_(float4 gl_FragCoord [[position]],
                   constant Globals& g [[buffer(0)]],
                   texture2d<float> iChannel0_tex [[texture(0)]],
                   sampler iChannel0_smp [[sampler(0)]]) {
    float2 fragCoord = float2(gl_FragCoord.x, g.iResolution.y - gl_FragCoord.y);
    float2 uv = fragCoord / float2(g.iResolution.x, g.iResolution.y);
    float4 scene = iChannel0_tex.sample(iChannel0_smp, uv);
    return Out{ float4(1.0 - scene.rgb, 1.0) };
}
"#;

        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);
        let frame = size(DevicePixels::from(256), DevicePixels::from(256));

        let red = Hsla {
            h: 0.0,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };
        let blue = Hsla {
            h: 240.0 / 360.0,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };

        let mut scene = Scene::default();
        // Full frame: top half red, bottom half blue.
        scene.insert_primitive(solid_quad(px_bounds(0.0, 0.0, 256.0, 128.0), red));
        scene.insert_primitive(solid_quad(px_bounds(0.0, 128.0, 256.0, 128.0), blue));
        // A full-frame pass, inserted LAST so it draws on top and captures the red/blue beneath it.
        let full = px_bounds(0.0, 0.0, 256.0, 256.0);
        // iResolution packed at offset 0 (packed_float3): the frame size in pixels.
        let mut uniforms = Vec::new();
        uniforms.extend_from_slice(&256.0f32.to_le_bytes());
        uniforms.extend_from_slice(&256.0f32.to_le_bytes());
        uniforms.extend_from_slice(&1.0f32.to_le_bytes());
        scene.insert_primitive(ShaderPass {
            order: 0,
            bounds: full,
            content_mask: ContentMask { bounds: full },
            corner_radii: Corners::default(),
            shader_id: 0xF11A,
            msl: INVERT_MSL.into(),
            fragment_entry: "main_".into(),
            uniforms: Arc::from(uniforms),
            samples_scene: true,
        });
        scene.finish();

        let rendered = renderer
            .render_scene_to_image(&scene, frame)
            .expect("render_scene_to_image should composite the scene-sampling pass");

        let is_cyan = |p: &image::Rgba<u8>| p.0[0] < 80 && p.0[1] > 180 && p.0[2] > 180;
        let is_yellow = |p: &image::Rgba<u8>| p.0[0] > 180 && p.0[1] > 180 && p.0[2] < 80;

        // Top quarter samples the frame's RED top and inverts it -> cyan. Yellow would mean the capture
        // was vertically flipped (sampled the blue bottom); red would mean the pass was skipped.
        let top = rendered.get_pixel(128, 64);
        let bottom = rendered.get_pixel(128, 192);
        assert!(
            is_cyan(top),
            "top should be inverted red (cyan) with an upright scene capture, got {top:?}"
        );
        assert!(
            is_yellow(bottom),
            "bottom should be inverted blue (yellow) with an upright scene capture, got {bottom:?}"
        );
    }

    fn quad(scene: &mut Scene, x: f32, y: f32, w: f32, h: f32, color: Hsla) {
        scene.insert_primitive(solid_quad(px_bounds(x, y, w, h), color));
    }

    /// Builds a mock coding-session view out of plain quads (no text system): a
    /// title bar with a "working" status dot, transcript lines, a syntax-tinted
    /// code block, and a prompt bar — enough silhouette that the shrunk-down
    /// miniature is recognizably the same view.
    fn build_mock_session_view(scene: &mut Scene) {
        let bg = Hsla {
            h: 0.62,
            s: 0.22,
            l: 0.12,
            a: 1.0,
        };
        let title = Hsla {
            h: 0.62,
            s: 0.40,
            l: 0.26,
            a: 1.0,
        };
        let working = Hsla {
            h: 0.58,
            s: 0.85,
            l: 0.55,
            a: 1.0,
        };
        let text = Hsla {
            h: 0.0,
            s: 0.0,
            l: 0.72,
            a: 1.0,
        };
        let dim = Hsla {
            h: 0.0,
            s: 0.0,
            l: 0.45,
            a: 1.0,
        };
        let user = Hsla {
            h: 0.62,
            s: 0.5,
            l: 0.30,
            a: 1.0,
        };
        let code_bg = Hsla {
            h: 0.62,
            s: 0.30,
            l: 0.07,
            a: 1.0,
        };
        let green = Hsla {
            h: 0.33,
            s: 0.55,
            l: 0.55,
            a: 1.0,
        };
        let orange = Hsla {
            h: 0.07,
            s: 0.70,
            l: 0.58,
            a: 1.0,
        };
        let purple = Hsla {
            h: 0.78,
            s: 0.45,
            l: 0.62,
            a: 1.0,
        };
        let prompt = Hsla {
            h: 0.62,
            s: 0.28,
            l: 0.17,
            a: 1.0,
        };

        quad(scene, 0.0, 0.0, 960.0, 720.0, bg);
        quad(scene, 0.0, 0.0, 960.0, 56.0, title);
        quad(scene, 24.0, 18.0, 20.0, 20.0, working);
        quad(scene, 60.0, 22.0, 220.0, 12.0, text);
        quad(scene, 32.0, 92.0, 540.0, 14.0, text);
        quad(scene, 32.0, 120.0, 620.0, 14.0, dim);
        quad(scene, 32.0, 148.0, 460.0, 14.0, dim);
        quad(scene, 32.0, 192.0, 720.0, 26.0, user);
        quad(scene, 44.0, 199.0, 360.0, 12.0, text);
        quad(scene, 32.0, 246.0, 896.0, 250.0, code_bg);
        quad(scene, 52.0, 268.0, 120.0, 12.0, purple);
        quad(scene, 184.0, 268.0, 240.0, 12.0, text);
        quad(scene, 72.0, 298.0, 180.0, 12.0, green);
        quad(scene, 268.0, 298.0, 320.0, 12.0, dim);
        quad(scene, 72.0, 328.0, 300.0, 12.0, orange);
        quad(scene, 72.0, 358.0, 220.0, 12.0, green);
        quad(scene, 304.0, 358.0, 180.0, 12.0, dim);
        quad(scene, 52.0, 388.0, 140.0, 12.0, purple);
        quad(scene, 72.0, 418.0, 380.0, 12.0, text);
        quad(scene, 72.0, 448.0, 260.0, 12.0, orange);
        quad(scene, 32.0, 520.0, 580.0, 14.0, text);
        quad(scene, 32.0, 548.0, 500.0, 14.0, dim);
        quad(scene, 32.0, 632.0, 896.0, 52.0, prompt);
        quad(scene, 48.0, 651.0, 16.0, 16.0, working);
        quad(scene, 76.0, 653.0, 300.0, 12.0, dim);
    }

    /// Opt-in visual artifact (set `KCODE_PROOF_DUMP=1`): renders the mock view
    /// once into a BGRA surface, then paints that *same* surface as a maximized
    /// panel plus a 2x2 grid of shrunk tiles, and writes the composite PNG. Every
    /// tile is the whole view uniformly scaled down — the Exposé/Mission-Control
    /// effect — proving it visually, not just by pixel assertions.
    #[test]
    fn dump_thumbnail_proof_png() {
        if std::env::var("KCODE_PROOF_DUMP").is_err() {
            return;
        }

        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);

        let view_size = size(DevicePixels::from(960), DevicePixels::from(720));
        let mut view = Scene::default();
        build_mock_session_view(&mut view);
        view.finish();
        let surface = renderer
            .render_scene_to_surface(&view, view_size)
            .expect("render mock session view to surface");

        let canvas = size(DevicePixels::from(1280), DevicePixels::from(640));
        let mut composed = Scene::default();
        composed.insert_primitive(solid_quad(
            px_bounds(0.0, 0.0, 1280.0, 640.0),
            Hsla {
                h: 0.62,
                s: 0.15,
                l: 0.05,
                a: 1.0,
            },
        ));
        let mut paint = |x: f32, y: f32, w: f32, h: f32| {
            let b = px_bounds(x, y, w, h);
            composed.insert_primitive(PaintSurface {
                order: 0,
                bounds: b,
                content_mask: ContentMask { bounds: b },
                image_buffer: surface.clone(),
            });
        };
        // Maximized panel (left) + a 2x2 grid of live miniatures (right), all 4:3.
        paint(40.0, 56.0, 700.0, 525.0);
        paint(780.0, 56.0, 230.0, 172.0);
        paint(1030.0, 56.0, 230.0, 172.0);
        paint(780.0, 250.0, 230.0, 172.0);
        paint(1030.0, 250.0, 230.0, 172.0);
        drop(paint);
        composed.finish();

        let image = renderer
            .render_scene_to_image(&composed, canvas)
            .expect("compose maximized panel + tiles");
        let out = "/tmp/kcode_m1_thumbnail_proof.png";
        image.save(out).expect("save proof png");
        println!("WROTE {out} ({}x{})", image.width(), image.height());
    }

    /// Regression smoke test for the YUV branch of `draw_surfaces` (the path the
    /// BGRA work refactored): build an NV12 (4:2:0 biplanar) surface with a known
    /// luma + neutral chroma, paint it scaled into a box, and assert the box
    /// renders a light gray while the surround stays black — proving the YUV
    /// branch still routes, samples both planes, and converts to RGB.
    #[test]
    fn paints_a_yuv_biplanar_surface_through_the_yuv_branch() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);

        // 64x64 NV12: Y=180 everywhere, Cb=Cr=128 (neutral) → a light gray.
        let (w, h) = (64usize, 64usize);
        let yuv =
            create_metal_surface_pixel_buffer(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, w, h)
                .expect("create nv12 buffer");
        unsafe {
            assert_eq!(yuv.lock_base_address(0), 0, "lock nv12 base address");
            // Plane 0: luma, full resolution, one byte per pixel.
            let y_ptr = yuv.get_base_address_of_plane(0) as *mut u8;
            let y_stride = yuv.get_bytes_per_row_of_plane(0);
            for row in 0..yuv.get_height_of_plane(0) {
                let line = y_ptr.add(row * y_stride);
                for col in 0..yuv.get_width_of_plane(0) {
                    *line.add(col) = 180;
                }
            }
            // Plane 1: chroma, half resolution, interleaved Cb,Cr (two bytes each).
            let c_ptr = yuv.get_base_address_of_plane(1) as *mut u8;
            let c_stride = yuv.get_bytes_per_row_of_plane(1);
            for row in 0..yuv.get_height_of_plane(1) {
                let line = c_ptr.add(row * c_stride);
                for col in 0..yuv.get_width_of_plane(1) {
                    *line.add(col * 2) = 128;
                    *line.add(col * 2 + 1) = 128;
                }
            }
            assert_eq!(yuv.unlock_base_address(0), 0, "unlock nv12 base address");
        }

        let frame = size(DevicePixels::from(128), DevicePixels::from(128));
        let box_bounds = px_bounds(32.0, 32.0, 64.0, 64.0);
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds: box_bounds,
            content_mask: ContentMask { bounds: box_bounds },
            image_buffer: yuv,
        });
        scene.finish();

        let rendered = renderer
            .render_scene_to_image(&scene, frame)
            .expect("render the painted nv12 surface");

        // Inside the box: a light, roughly-neutral gray.
        let inside = rendered.get_pixel(64, 64);
        let [r, g, b, _] = inside.0;
        let lo = r.min(g).min(b);
        let hi = r.max(g).max(b);
        assert!(lo > 100, "box should be a light gray, got {inside:?}");
        assert!(
            hi - lo < 50,
            "box should be roughly neutral, got {inside:?}"
        );
        // Outside the box stays black.
        let outside = rendered.get_pixel(8, 8);
        assert!(
            outside.0[0] < 40 && outside.0[1] < 40 && outside.0[2] < 40,
            "outside the box should be black, got {outside:?}"
        );
    }
}
