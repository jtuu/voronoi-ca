use bytemuck::{Pod, Zeroable};

pub fn hsv2rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    _pos: [f32; 4],
    _color: [f32; 3]
}

pub fn vertex(x: f64, y: f64, r: f64, g: f64, b: f64) -> Vertex {
    Vertex {
        _pos: [x as f32, y as f32, 0.0, 1.0],
        _color: [r as f32, g as f32, b as f32],
    }
}

pub fn generate_matrix(width: f32, height: f32) -> glam::Mat4 {
    let aspect1 = width / height * 2.0;
    let aspect2 = height / width * 2.0;
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::ZERO,
        glam::Vec3::new(0.0, 0.0, 1.0),
        glam::Vec3::Y
    );
    let dim = 0.75;
    let projection =  glam::Mat4::orthographic_rh(-dim, dim, -dim, dim, 0.0, 1000.0);
    return view * projection;
}

pub fn feq(a: f64, b: f64) -> bool {
    return a >= b - f64::EPSILON && a <= b + f64::EPSILON;
}
