use std::{borrow::Cow, mem};
use bytemuck::{Pod, Zeroable};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu::util::DeviceExt;
use rand::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _color: [f32; 3]
}

fn vertex(x: f32, y: f32, r: f32, g: f32, b: f32) -> Vertex {
    Vertex {
        _pos: [x, y, 0.0, 1.0],
        _color: [r, g, b],
    }
}

fn generate_matrix(width: f32, height: f32) -> glam::Mat4 {
    let aspect1 = width / height * 2.0;
    let aspect2 = height / width * 2.0;
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::ZERO,
        glam::Vec3::new(0.0, 0.0, 1.0),
        glam::Vec3::Y
    );
    let projection =  glam::Mat4::orthographic_rh(-aspect1, aspect1, -aspect2, aspect2, 0.0, 1000.0);
    return view * projection;
}

fn generate_grid() -> Vec<(f64, f64)> {
    let mut points = Vec::new();
    let dim = 10;
    for x in -dim..dim {
        for y in -dim..dim {
            points.push((x as f64 / dim as f64, y as f64 / dim as f64));
        }
    }
    return points;
}

fn generate_points() -> Vec<(f64, f64)> {
    let mut points = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        points.push((rng.gen(), rng.gen()));
    }

    return points;
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mut rng = rand::thread_rng();

    let voronoi = voronator::VoronoiDiagram::<voronator::delaunator::Point>::from_tuple(&(-1.0, -1.0), &(1.0, 1.0), &generate_points()).unwrap();
    for cell in voronoi.cells() {
        let points = cell.points();
        let diagram = voronator::CentroidDiagram::<voronator::delaunator::Point>::new(points).unwrap();
        let c: f32 = rng.gen();
        for tri_idx in diagram.delaunay.triangles {
            let p = &points[tri_idx];
            vertices.push(vertex(p.x as f32, p.y as f32, c, 0.0, 0.0));
        }
    }

    for i in 0..vertices.len() {
        indices.push(i as u16);
    }

    return (vertices, indices);
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::default(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");
    
    let vertex_size = mem::size_of::<Vertex>();
    let (vertex_data, index_data) = create_vertices();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    let vertex_buffers = [wgpu::VertexBufferLayout {
        array_stride: vertex_size as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 4 * 4,
                shader_location: 1,
            },
        ],
    }];

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // Create pipeline layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create other resources
    let mx_total = generate_matrix(size.width as f32, size.height as f32);
    let mx_ref: &[f32; 16] = mx_total.as_ref();
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(mx_ref),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }
        ],
        label: None,
    });

    let swapchain_format = surface.get_supported_formats(&adapter)[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
    };

    surface.configure(&device, &config);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    input: winit::event::KeyboardInput {
                        virtual_keycode: Some(keycode),
                        state,
                        ..
                    },
                    ..
                },
                ..
            } => {
                match state {
                    winit::event::ElementState::Pressed => {
                        if keycode == winit::event::VirtualKeyCode::Escape {
                            *control_flow = ControlFlow::Exit
                        }
                    },
                    _ => {}
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint16);
                    rpass.set_vertex_buffer(0, vertex_buf.slice(..));
                    rpass.draw_indexed(0 .. index_data.len() as u32, 0, 0 .. (vertex_data.len() / 3) as u32);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        // Temporarily avoid srgb formats for the swapchain on the web
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
