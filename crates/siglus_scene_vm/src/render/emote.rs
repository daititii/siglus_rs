use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use eluna::{EmoteDrawFrameInfo, EmoteDrawPass, EmoteStaticScene, EmoteStaticSprite};
use wgpu::util::DeviceExt;

use crate::emote::EmoteRenderPacket;

use super::GpuTexture;

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

const SHADER: &str = r#"
@group(0) @binding(0) var sprite_tex: texture_2d<f32>;
@group(0) @binding(1) var sprite_sampler: sampler;

struct VertexIn {
    @location(0) clip_position: vec2<f32>,
    @location(1) model_position: vec2<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) blend_mode: f32,
    @location(5) clip_rect: vec4<f32>,
    @location(6) wipe: vec3<f32>,
};
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) model_position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) blend_mode: f32,
    @location(4) clip_rect: vec4<f32>,
    @location(5) wipe: vec3<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(input.clip_position, 0.0, 1.0);
    out.model_position = input.model_position;
    out.texcoord = input.texcoord;
    out.color = input.color;
    out.blend_mode = input.blend_mode;
    out.clip_rect = input.clip_rect;
    out.wipe = input.wipe;
    return out;
}

fn native_texture_stage(input_c: vec4<f32>, blend_mode: u32) -> vec4<f32> {
    var rgb = input_c.rgb;
    let alpha = input_c.a;
    if ((blend_mode & 0xF0u) == 0x10u) {
        rgb = clamp(rgb * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let low = blend_mode & 0xFF0Fu;
    if (low == 3u || low == 4u) {
        rgb = rgb * alpha;
    } else if (low == 5u) {
        rgb = vec3<f32>(1.0) - rgb;
    }
    return vec4<f32>(rgb, alpha);
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    if (input.model_position.x < input.clip_rect.x || input.model_position.y < input.clip_rect.y ||
        input.model_position.x > input.clip_rect.z || input.model_position.y > input.clip_rect.w) {
        discard;
    }
    var c = textureSample(sprite_tex, sprite_sampler, input.texcoord) * input.color;
    if (input.wipe.z > 0.5) {
        c = vec4<f32>(c.rgb, clamp(c.a * input.wipe.x + input.wipe.y, 0.0, 1.0));
    }
    c = native_texture_stage(c, u32(input.blend_mode + 0.5));
    if (c.a <= 0.003) {
        discard;
    }
    return c;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmoteVertex {
    clip_position: [f32; 2],
    model_position: [f32; 2],
    texcoord: [f32; 2],
    color: [f32; 4],
    blend_mode: f32,
    clip_rect: [f32; 4],
    wipe: [f32; 3],
}

impl EmoteVertex {
    const ATTRS: [wgpu::VertexAttribute; 7] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 24, shader_location: 3 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 40, shader_location: 4 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 44, shader_location: 5 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 60, shader_location: 6 },
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[derive(Debug)]
struct Target {
    output: GpuTexture,
    feedback: GpuTexture,
    feedback_valid: bool,
    alpha_readback_version: Option<u64>,
    stencil_texture: wgpu::Texture,
    stencil_view: wgpu::TextureView,
    textures: HashMap<u32, GpuTexture>,
    texture_bind_groups: HashMap<u32, wgpu::BindGroup>,
    feedback_bind_group: wgpu::BindGroup,
    version: u64,
}

#[derive(Debug)]
struct Draw {
    texture: DrawTexture,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    blend_index: usize,
    stencil_groups: Vec<StencilGroup>,
    stencil_initial_reference: u32,
    stencil_final_reference: u32,
}

#[derive(Debug, Clone, Copy)]
enum DrawTexture {
    Resource(u32),
    Feedback,
}

#[derive(Debug)]
struct StencilSource {
    texture: DrawTexture,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

#[derive(Debug)]
struct StencilGroup {
    phase: u32,
    sources: Vec<StencilSource>,
}

#[derive(Debug)]
pub(super) struct EmoteCompositor {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    color_pipelines: Vec<wgpu::RenderPipeline>,
    stencil_color_pipelines: Vec<wgpu::RenderPipeline>,
    stencil_inner_pipeline: wgpu::RenderPipeline,
    stencil_outer_pipeline: wgpu::RenderPipeline,
    targets: HashMap<u64, Target>,
}

impl EmoteCompositor {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("siglus-emote-texture-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("siglus-emote-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("siglus-emote-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let color_pipelines = (0..6)
            .map(|mode| create_color_pipeline(device, &pipeline_layout, &shader, mode, false))
            .collect();
        let stencil_color_pipelines = (0..6)
            .map(|mode| create_color_pipeline(device, &pipeline_layout, &shader, mode, true))
            .collect();
        let stencil_inner_pipeline = create_mask_pipeline(
            device,
            &pipeline_layout,
            &shader,
            wgpu::StencilOperation::IncrementWrap,
            "siglus-emote-inner-mask",
        );
        let stencil_outer_pipeline = create_mask_pipeline(
            device,
            &pipeline_layout,
            &shader,
            wgpu::StencilOperation::DecrementWrap,
            "siglus-emote-outer-mask",
        );
        Self {
            bind_group_layout,
            pipeline_layout,
            shader,
            color_pipelines,
            stencil_color_pipelines,
            stencil_inner_pipeline,
            stencil_outer_pipeline,
            targets: HashMap::new(),
        }
    }

    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &EmoteRenderPacket,
    ) -> Result<()> {
        let recreate = self.targets.get(&packet.render_id).map_or(true, |target| {
            target.output.width != packet.width || target.output.height != packet.height
        });
        if recreate {
            let target = create_target(device, queue, &self.bind_group_layout, packet)?;
            self.targets.insert(packet.render_id, target);
        }
        // Drive map_async callbacks without blocking. This is also valid on
        // wasm32, where Maintain::Wait is not available as a synchronous path.
        device.poll(wgpu::Maintain::Poll);
        if self
            .targets
            .get(&packet.render_id)
            .is_some_and(|target| target.version == packet.version)
        {
            if packet.alpha_readback && !packet.has_current_hit_surface() {
                let target = self
                    .targets
                    .get_mut(&packet.render_id)
                    .ok_or_else(|| anyhow!("Emote compositor target disappeared"))?;
                if target.alpha_readback_version != Some(packet.version) {
                    schedule_alpha_readback(device, queue, target, packet);
                    target.alpha_readback_version = Some(packet.version);
                }
            }
            return Ok(());
        }

        let draws = build_draws(device, packet)?;
        let target = self
            .targets
            .get(&packet.render_id)
            .ok_or_else(|| anyhow!("Emote compositor target disappeared"))?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("siglus-emote-object-encoder"),
        });
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("siglus-emote-object-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        for draw in &draws {
            let Some(bind_group) = resolve_bind_group(target, draw.texture) else {
                continue;
            };
            if draw.stencil_groups.is_empty() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("siglus-emote-color"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target.output.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.color_pipelines[draw.blend_index]);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.draw(0..draw.vertex_count, 0..1);
                continue;
            }

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("siglus-emote-stencil-color"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.stencil_view,
                    depth_ops: None,
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(draw.stencil_initial_reference),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut reference = draw.stencil_initial_reference;
            for group in draw.stencil_groups.iter().filter(|group| group.phase == 1) {
                pass.set_pipeline(&self.stencil_inner_pipeline);
                pass.set_stencil_reference(reference);
                for source in &group.sources {
                    let Some(mask_bind) = resolve_bind_group(target, source.texture) else {
                        continue;
                    };
                    pass.set_bind_group(0, mask_bind, &[]);
                    pass.set_vertex_buffer(0, source.vertex_buffer.slice(..));
                    pass.draw(0..source.vertex_count, 0..1);
                }
                reference = reference.saturating_add(1).min(255);
            }
            let final_reference = draw.stencil_final_reference;
            if draw.stencil_groups.iter().any(|group| group.phase == 2) {
                pass.set_pipeline(&self.stencil_outer_pipeline);
                pass.set_stencil_reference(final_reference);
                for group in draw.stencil_groups.iter().filter(|group| group.phase == 2) {
                    for source in &group.sources {
                        let Some(mask_bind) = resolve_bind_group(target, source.texture) else {
                            continue;
                        };
                        pass.set_bind_group(0, mask_bind, &[]);
                        pass.set_vertex_buffer(0, source.vertex_buffer.slice(..));
                        pass.draw(0..source.vertex_count, 0..1);
                    }
                }
            }
            pass.set_pipeline(&self.stencil_color_pipelines[draw.blend_index]);
            pass.set_stencil_reference(final_reference);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &target.output._tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &target.feedback._tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: packet.width,
                height: packet.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let target = self.targets.get_mut(&packet.render_id).unwrap();
        target.feedback_valid = true;
        target.version = packet.version;
        if packet.alpha_readback && target.alpha_readback_version != Some(packet.version) {
            schedule_alpha_readback(device, queue, target, packet);
            target.alpha_readback_version = Some(packet.version);
        }
        Ok(())
    }

    pub(super) fn texture(&self, render_id: u64) -> Option<&GpuTexture> {
        self.targets.get(&render_id).map(|target| &target.output)
    }

    pub(super) fn retain_render_ids(&mut self, live: &HashSet<u64>) {
        self.targets.retain(|render_id, _| live.contains(render_id));
    }
}

fn schedule_alpha_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &Target,
    packet: &EmoteRenderPacket,
) {
    let width = packet.width.max(1);
    let height = packet.height.max(1);
    let unpadded_bytes_per_row = width.saturating_mul(4);
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("siglus-emote-alpha-readback"),
        size: padded_bytes_per_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("siglus-emote-alpha-readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &target.output._tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: buffer.as_ref(),
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let callback_buffer = buffer.clone();
    let callback_packet = packet.clone();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        if let Err(err) = result {
            log::error!("Emote alpha readback failed: {err}");
            return;
        }
        let width = callback_packet.width as usize;
        let height = callback_packet.height as usize;
        let row_bytes = width.saturating_mul(4);
        let data = callback_buffer.slice(..).get_mapped_range();
        let mut alpha = vec![0u8; width.saturating_mul(height)];
        for y in 0..height {
            let src_offset = y.saturating_mul(padded_bytes_per_row as usize);
            let src_end = src_offset.saturating_add(row_bytes);
            if src_end > data.len() {
                break;
            }
            for x in 0..width {
                alpha[y * width + x] = data[src_offset + x * 4 + 3];
            }
        }
        drop(data);
        callback_buffer.unmap();
        callback_packet.publish_hit_alpha(alpha);
    });
}

fn create_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    packet: &EmoteRenderPacket,
) -> Result<Target> {
    let output = create_gpu_texture(
        device,
        "siglus-emote-object-output",
        packet.width,
        packet.height,
        TARGET_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        0,
    );
    let feedback = create_gpu_texture(
        device,
        "siglus-emote-feedback",
        packet.width,
        packet.height,
        TARGET_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        0,
    );

    // eng_emote.cpp sets both D3DSAMP_MAGFILTER and D3DSAMP_MINFILTER to
    // D3DTEXF_POINT while the Emote player renders into its object RT. Keep
    // that internal sampler separate from `GpuTexture::sampler`: the latter is
    // consumed later by the ordinary Siglus OBJECT renderer, where the Emote
    // RT behaves like any other object texture.
    let internal_point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("siglus-emote-internal-point-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let feedback_bind_group =
        create_texture_bind_group(device, layout, &feedback, &internal_point_sampler);

    let stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("siglus-emote-stencil"),
        size: wgpu::Extent3d {
            width: packet.width,
            height: packet.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: STENCIL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let stencil_view = stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut textures = HashMap::new();
    let mut texture_bind_groups = HashMap::new();
    for (&resource_index, source) in packet.textures.iter() {
        let tex = create_gpu_texture(
            device,
            "siglus-emote-resource",
            source.width,
            source.height,
            TARGET_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            0,
        );
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex._tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            source.rgba.as_slice(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * source.width),
                rows_per_image: Some(source.height),
            },
            wgpu::Extent3d {
                width: source.width,
                height: source.height,
                depth_or_array_layers: 1,
            },
        );
        let bind_group =
            create_texture_bind_group(device, layout, &tex, &internal_point_sampler);
        textures.insert(resource_index, tex);
        texture_bind_groups.insert(resource_index, bind_group);
    }

    Ok(Target {
        output,
        feedback,
        feedback_valid: false,
        alpha_readback_version: None,
        stencil_texture,
        stencil_view,
        textures,
        texture_bind_groups,
        feedback_bind_group,
        version: 0,
    })
}

fn create_gpu_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    version: u64,
) -> GpuTexture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("siglus-emote-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    GpuTexture {
        _tex: tex,
        view,
        sampler,
        width: width.max(1),
        height: height.max(1),
        version,
    }
}

fn create_texture_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture: &GpuTexture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("siglus-emote-texture-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn resolve_bind_group<'a>(target: &'a Target, texture: DrawTexture) -> Option<&'a wgpu::BindGroup> {
    match texture {
        DrawTexture::Resource(index) => target.texture_bind_groups.get(&index),
        DrawTexture::Feedback => target.feedback_valid.then_some(&target.feedback_bind_group),
    }
}

fn native_blend_state(mode: u32) -> wgpu::BlendState {
    let (operation, src_factor, dst_factor) = match mode {
        1 => (wgpu::BlendOperation::Add, wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::One),
        2 | 5 => (
            wgpu::BlendOperation::ReverseSubtract,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::One,
        ),
        3 => (
            wgpu::BlendOperation::Add,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusSrcAlpha,
        ),
        4 => (
            wgpu::BlendOperation::Add,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::One,
        ),
        _ => (
            wgpu::BlendOperation::Add,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
        ),
    };
    let color = wgpu::BlendComponent { src_factor, dst_factor, operation };
    let alpha = if mode == 0 {
        wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        }
    } else {
        wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        }
    };
    wgpu::BlendState { color, alpha }
}

fn native_blend_index(blend_mode: u32) -> usize {
    match blend_mode & 0xFF0F {
        0..=5 => (blend_mode & 0xFF0F) as usize,
        _ => 0,
    }
}

fn create_color_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    mode: u32,
    stencil: bool,
) -> wgpu::RenderPipeline {
    let depth_stencil = stencil.then_some(wgpu::DepthStencilState {
        format: STENCIL_FORMAT,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState {
            front: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            },
            back: wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            },
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: wgpu::DepthBiasState::default(),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("siglus-emote-color-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_main",
            buffers: &[EmoteVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: Some(native_blend_state(mode)),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

fn create_mask_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    pass_op: wgpu::StencilOperation,
    label: &str,
) -> wgpu::RenderPipeline {
    let face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: pass_op,
        pass_op,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_main",
            buffers: &[EmoteVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: STENCIL_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: face,
                back: face,
                read_mask: 0xff,
                write_mask: 0xff,
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

fn build_draws(device: &wgpu::Device, packet: &EmoteRenderPacket) -> Result<Vec<Draw>> {
    let scene = packet.scene.as_ref();
    let visible: Vec<&EmoteStaticSprite> = scene
        .sprites
        .iter()
        .filter(|sprite| sprite.visible && sprite.opacity > 0.0)
        .collect();
    let mut key_to_sprite = HashMap::<Vec<u64>, &EmoteStaticSprite>::new();
    for sprite in &visible {
        key_to_sprite.insert(sprite.draw_frame_info.native_draw_key.clone(), *sprite);
    }
    let layer_infos: HashMap<Vec<u64>, &EmoteDrawFrameInfo> = scene
        .layer_states
        .iter()
        .map(|state| (state.draw_frame_info.native_draw_key.clone(), &state.draw_frame_info))
        .collect();

    let mut draws = Vec::new();
    for sprite in visible {
        if matches!(
            sprite.draw_frame_info.pass,
            EmoteDrawPass::MaskGeneration | EmoteDrawPass::StencilCompositeMask
        ) {
            continue;
        }
        if !sprite.feedback_history && !packet.textures.contains_key(&sprite.texture_resource_index) {
            continue;
        }
        let (stencil_groups, initial_reference, final_reference) = stencil_groups_for_sprite(
            device,
            packet,
            scene,
            &key_to_sprite,
            &layer_infos,
            sprite,
        )?;
        let vertices = sprite_vertices(packet, sprite);
        if vertices.is_empty() {
            continue;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("siglus-emote-draw-vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        draws.push(Draw {
            texture: if sprite.feedback_history {
                DrawTexture::Feedback
            } else {
                DrawTexture::Resource(sprite.texture_resource_index)
            },
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            blend_index: native_blend_index(sprite.blend_mode),
            stencil_groups,
            stencil_initial_reference: initial_reference,
            stencil_final_reference: final_reference,
        });
    }
    Ok(draws)
}

fn stencil_groups_for_sprite(
    device: &wgpu::Device,
    packet: &EmoteRenderPacket,
    scene: &EmoteStaticScene,
    key_to_sprite: &HashMap<Vec<u64>, &EmoteStaticSprite>,
    layer_infos: &HashMap<Vec<u64>, &EmoteDrawFrameInfo>,
    sprite: &EmoteStaticSprite,
) -> Result<(Vec<StencilGroup>, u32, u32)> {
    let mut chain = Vec::<&EmoteDrawFrameInfo>::new();
    let mut cursor = sprite.draw_frame_info.stencil_parent_native_key.as_ref();
    let mut visited = HashSet::<Vec<u64>>::new();
    while let Some(key) = cursor {
        if !visited.insert(key.clone()) {
            break;
        }
        let Some(info) = layer_infos.get(key).copied() else {
            break;
        };
        if matches!(info.stencil_phase, 1 | 2) {
            chain.push(info);
        }
        cursor = info.stencil_parent_native_key.as_ref();
    }
    if chain.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }
    let initial_reference: u32 = if chain.iter().any(|info| info.stencil_phase == 2) { 1 } else { 0 };
    let mut final_reference = initial_reference;
    let mut groups = Vec::new();
    for phase in [1i64, 2i64] {
        for info in chain.iter().copied().filter(|info| info.stencil_phase == phase) {
            let source_keys = if (info.stencil_type & 4) != 0 {
                scene
                    .composite_mask_sources_by_key
                    .get(&info.native_draw_key)
                    .cloned()
                    .unwrap_or_default()
            } else {
                vec![info.native_draw_key.clone()]
            };
            let mut sources = Vec::new();
            for key in source_keys {
                let Some(source) = key_to_sprite.get(&key).copied() else {
                    continue;
                };
                if !source.feedback_history && !packet.textures.contains_key(&source.texture_resource_index) {
                    continue;
                }
                let vertices = sprite_vertices(packet, source);
                if vertices.is_empty() {
                    continue;
                }
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("siglus-emote-stencil-vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                sources.push(StencilSource {
                    texture: if source.feedback_history {
                        DrawTexture::Feedback
                    } else {
                        DrawTexture::Resource(source.texture_resource_index)
                    },
                    vertex_buffer,
                    vertex_count: vertices.len() as u32,
                });
            }
            groups.push(StencilGroup { phase: phase as u32, sources });
            if phase == 1 {
                final_reference = final_reference.saturating_add(1).min(255);
            }
        }
    }
    Ok((groups, initial_reference, final_reference))
}

fn sprite_vertices(packet: &EmoteRenderPacket, sprite: &EmoteStaticSprite) -> Vec<EmoteVertex> {
    if let Some(mesh) = &sprite.mesh {
        return mesh_sprite_vertices(packet, sprite, mesh);
    }
    let left = sprite.left();
    let right = sprite.right();
    let top = sprite.top();
    let bottom = sprite.bottom();
    let tl = native_corner_color(sprite, sprite.corner_colors[0]);
    let tr = native_corner_color(sprite, sprite.corner_colors[1]);
    let bl = native_corner_color(sprite, sprite.corner_colors[2]);
    let br = native_corner_color(sprite, sprite.corner_colors[3]);
    let make = |position: [f32; 2], texcoord: [f32; 2], color: [f32; 4]| {
        make_vertex(packet, sprite, transform_sprite_point(sprite, position), texcoord, color)
    };
    vec![
        make([left, top], [sprite.uv_left, sprite.uv_top], tl),
        make([left, bottom], [sprite.uv_left, sprite.uv_bottom], bl),
        make([right, top], [sprite.uv_right, sprite.uv_top], tr),
        make([right, top], [sprite.uv_right, sprite.uv_top], tr),
        make([left, bottom], [sprite.uv_left, sprite.uv_bottom], bl),
        make([right, bottom], [sprite.uv_right, sprite.uv_bottom], br),
    ]
}

fn mesh_sprite_vertices(
    packet: &EmoteRenderPacket,
    sprite: &EmoteStaticSprite,
    mesh: &eluna::EmoteMeshPatch,
) -> Vec<EmoteVertex> {
    let division_x = mesh.division_x.max(1) as usize;
    let division_y = mesh.division_y.max(1) as usize;
    let left = sprite.left();
    let top = sprite.top();
    let corner_colors = [
        native_corner_color(sprite, sprite.corner_colors[0]),
        native_corner_color(sprite, sprite.corner_colors[1]),
        native_corner_color(sprite, sprite.corner_colors[2]),
        native_corner_color(sprite, sprite.corner_colors[3]),
    ];
    let vertex_at = |ix: usize, iy: usize| {
        let u = ix as f32 / division_x as f32;
        let v = iy as f32 / division_y as f32;
        let p = mesh.sample(u, v);
        let position = [left + p[0] * sprite.width, top + p[1] * sprite.height];
        make_vertex(
            packet,
            sprite,
            transform_sprite_point(sprite, position),
            [
                sprite.uv_left + (sprite.uv_right - sprite.uv_left) * u,
                sprite.uv_top + (sprite.uv_bottom - sprite.uv_top) * v,
            ],
            bilerp_color(corner_colors, u, v),
        )
    };
    let mut vertices = Vec::with_capacity(division_x * division_y * 6);
    for y in 0..division_y {
        for x in 0..division_x {
            let tl = vertex_at(x, y);
            let bl = vertex_at(x, y + 1);
            let tr = vertex_at(x + 1, y);
            let br = vertex_at(x + 1, y + 1);
            vertices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    vertices
}

fn make_vertex(
    packet: &EmoteRenderPacket,
    sprite: &EmoteStaticSprite,
    model_position: [f32; 2],
    texcoord: [f32; 2],
    color: [f32; 4],
) -> EmoteVertex {
    // Original Siglus D3D9 Emote render target transform:
    // world translation (-rep_x,+rep_y), half-pixel-adjusted orthographic projection.
    let clip_x = 2.0 * (model_position[0] - packet.rep_x - 0.5) / packet.width.max(1) as f32;
    let clip_y = 2.0 * (0.5 - (model_position[1] + packet.rep_y)) / packet.height.max(1) as f32;
    EmoteVertex {
        clip_position: [clip_x, clip_y],
        model_position,
        texcoord,
        color,
        blend_mode: sprite.blend_mode as f32,
        clip_rect: sprite
            .draw_frame_info
            .clip_rect
            .unwrap_or([-1.0e30, -1.0e30, 1.0e30, 1.0e30]),
        wipe: [
            sprite.draw_frame_info.stencil_wipe_scale,
            sprite.draw_frame_info.stencil_wipe_bias,
            if sprite.draw_frame_info.stencil_wipe_enabled { 1.0 } else { 0.0 },
        ],
    }
}

fn native_corner_color(sprite: &EmoteStaticSprite, packed: u32) -> [f32; 4] {
    let r = ((packed >> 24) & 0xff) as f32 / 255.0;
    let g = ((packed >> 16) & 0xff) as f32 / 255.0;
    let b = ((packed >> 8) & 0xff) as f32 / 255.0;
    let a = (packed & 0xff) as f32 / 255.0;
    [r, g, b, (a * sprite.opacity).clamp(0.0, 1.0)]
}

fn bilerp_color(corners: [[f32; 4]; 4], u: f32, v: f32) -> [f32; 4] {
    let mut out = [0.0; 4];
    for i in 0..4 {
        let top = corners[0][i] + (corners[1][i] - corners[0][i]) * u;
        let bottom = corners[2][i] + (corners[3][i] - corners[2][i]) * u;
        out[i] = top + (bottom - top) * v;
    }
    out
}

fn transform_sprite_point(sprite: &EmoteStaticSprite, point: [f32; 2]) -> [f32; 2] {
    let sx = if sprite.scale_x.is_finite() { sprite.scale_x } else { 1.0 };
    let sy = if sprite.scale_y.is_finite() { sprite.scale_y } else { 1.0 };
    let angle = if sprite.rotation_degrees.is_finite() {
        sprite.rotation_degrees.to_radians()
    } else {
        0.0
    };
    let (sin, cos) = angle.sin_cos();
    let dx = (point[0] - sprite.center_x) * sx;
    let dy = (point[1] - sprite.center_y) * sy;
    let local = [
        sprite.center_x + dx * cos - dy * sin,
        sprite.center_y + dx * sin + dy * cos,
    ];
    let m = sprite.world_transform;
    [
        m[0] * local[0] + m[1] * local[1] + m[4],
        m[2] * local[0] + m[3] * local[1] + m[5],
    ]
}
