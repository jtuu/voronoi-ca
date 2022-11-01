use std::{borrow::Cow, mem, boxed::Box};
use bytemuck::{Pod, Zeroable};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use wgpu::util::DeviceExt;
use rand::prelude::*;

fn hsv2rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let hh = (h % 360.0) / 60.0;

    let i = hh as usize;
    let ff = hh - hh.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - (s * ff));
    let t = v * (1.0 - (s * (1.0 - ff)));

    return match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q)
    };
}

#[derive(Clone, Copy)]
struct Cell {
    alive: bool,
    x: f64,
    y: f64,
}

impl Cell {
    fn new(x: f64, y: f64) -> Self {
        return Self {
            x, y,
            alive: false
        };
    }

    fn neighbors(&self) -> Vec<Cell> {
        return vec![];
    }
}

struct CellularAutomaton {
    read: Vec<Cell>,
    write: Vec<Cell>
}

impl CellularAutomaton {
    fn new() -> Self {
        let cells = Vec::new();
        for p in generate_points() {

        }
        return Self {
            write: Vec::with_capacity(cells.len()),
            read: cells,
        };
    }

    fn step(&mut self) {
        for i in 0..self.read.len() {
            let read = &self.read[i];
            let write = &mut self.write[i];
            let mut neighbors = 0;
            for neigh in read.neighbors()  {
                if neigh.alive {
                    neighbors += 1;
                }
            }
            if read.alive {
                if neighbors < 2 {
                    write.alive = false;
                } else if neighbors == 2 || neighbors == 3 {
                    write.alive = true;
                } else {
                    write.alive = false;
                }
            } else {
                if neighbors == 3 {
                    write.alive = true;
                } else {
                    write.alive = false;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Point {
    ox: f64,
    oy: f64,
    x: f64,
    y: f64,
    r: f64,
    g: f64,
    b: f64
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _color: [f32; 3]
}

fn vertex(x: f64, y: f64, r: f64, g: f64, b: f64) -> Vertex {
    Vertex {
        _pos: [x as f32, y as f32, 0.0, 1.0],
        _color: [r as f32, g as f32, b as f32],
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

fn generate_points() -> Vec<Point> {
    let mut points = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let c = if rng.gen() {
            hsv2rgb(rng.gen_range(0.0..20.0), 0.9, 0.5 + rng.gen_range(0.0..0.5))
        } else {
            hsv2rgb(270.0 + rng.gen_range(0.0..20.0), 0.5 + rng.gen_range(0.0..0.2), 0.1 + rng.gen_range(0.0..0.9))
        };
        points.push(Point {
            ox: rng.gen(),
            oy: rng.gen(),
            x: 0.0,
            y: 0.0,
            r: c.0,
            g: c.1,
            b: c.2
        });
    }

    return points;
}

fn create_vertices(points: &[Point]) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Voronoify input
    let sites: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
    let voronoi = voronator::VoronoiDiagram::<voronator::delaunator::Point>::from_tuple(
        &(-1.0, -1.0), &(1.0, 1.0), &sites).unwrap();

    // Iterate over voronoi cells
    for (cell_idx, cell) in voronoi.cells().iter().enumerate() {
        // Create vertices by triangulating cells
        let cell_points = cell.points();

        if let Some(diagram) = voronator::CentroidDiagram::<voronator::delaunator::Point>::new(cell_points) {
            // Get color from original site point
            let site_p = &points[cell_idx];
    
            // Iterate over cell's triangles
            for tri_idx in diagram.delaunay.triangles {
                let tri_p = &cell_points[tri_idx];
                vertices.push(vertex(tri_p.x, tri_p.y,
                                     site_p.r, site_p.g, site_p.b));
            }
        }
    }

    indices.reserve_exact(vertices.len());
    for i in 0..vertices.len() {
        indices.push(i as u32);
    }

    return (vertices, indices);
}

fn animate(points: &mut [Point]) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap().as_millis() as f64;
        
    for point in points {
        let sin = ((now / 1000000.0 + point.ox) * 100.0).sin();
        let cos = ((now / 1000000.0 + point.oy) * 100.0).cos();
        point.x = point.ox + sin;
        point.y = point.oy + cos;
    }
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
    
    let vertex_size = mem::size_of::<Vertex>() as wgpu::BufferAddress;
    let mut points = generate_points();
    let max_vert_count = points.len() * 100; // Rough estimate of maximum number of vertices

    let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Buffer"),
        size: (max_vert_count * mem::size_of::<Vertex>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
    });

    let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Index Buffer"),
        size: (max_vert_count * mem::size_of::<u32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
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

    #[cfg(not(target_arch = "wasm32"))]
    let mut last_frame_inst = std::time::Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let (mut frame_count, mut accum_time) = (0, 0.0);

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
            Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    accum_time += last_frame_inst.elapsed().as_secs_f32();
                    last_frame_inst = std::time::Instant::now();
                    frame_count += 1;
                    if frame_count == 100 {
                        println!(
                            "Avg frame time {}ms",
                            accum_time * 1000.0 / frame_count as f32
                        );
                        accum_time = 0.0;
                        frame_count = 0;
                    }
                }

                // Update vertices
                animate(&mut points);
                let (vertex_data, index_data) = create_vertices(&points);
                let ind_bytes = bytemuck::cast_slice(&index_data);
                let vert_bytes = bytemuck::cast_slice(&vertex_data);
                queue.write_buffer(&index_buf, 0, ind_bytes);
                queue.write_buffer(&vertex_buf, 0, vert_bytes);

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
                    rpass.set_index_buffer(index_buf.slice(0..ind_bytes.len() as wgpu::BufferAddress), wgpu::IndexFormat::Uint32);
                    rpass.set_vertex_buffer(0, vertex_buf.slice(0..vert_bytes.len() as wgpu::BufferAddress));
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
