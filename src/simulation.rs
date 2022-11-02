use rand::prelude::*;
use super::common::*;

// Doesn't take into account the cell's current state
type Rule = Vec<(f64, bool)>;
// Rule2 applies a different rule based on the cell's current state (Born/Survive)
type Rule2 = [Vec<(f64, bool)>; 2];
type HSV = (f64, f64, f64);
type RGB = (f64, f64, f64);
type VoronoiDiagram = voronator::VoronoiDiagram::<voronator::delaunator::Point>;

#[derive(Clone, Copy, Debug)]
struct Cell {
    ox: f64,
    oy: f64,
    rx: f64,
    ry: f64,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    ocolor: HSV, 
    color: RGB,
    state: [bool; 2]
}

impl Cell {
    fn set_state(&mut self, i: usize, s: bool) {
        self.state[i] = s;
        // Desaturate color when dead
        self.color = if s {
            hsv2rgb(self.ocolor.0, self.ocolor.1, self.ocolor.2)
        } else {
            hsv2rgb(self.ocolor.0, 0.0, self.ocolor.2)
        };
    }

    fn new(x: f64, y: f64, alive: bool) -> Self {
        let mut rng = rand::thread_rng();
        // Random color
        let c = if rng.gen() {
            // Red
            (rng.gen_range(0.0..20.0), 0.9, 0.5 + rng.gen_range(0.0..0.3))
        } else {
            // Blue
            (270.0 + rng.gen_range(0.0..20.0), 0.5 + rng.gen_range(0.0..0.2), 0.3 + rng.gen_range(0.0..0.2))
        };

        let mut s = Self {
            ox: x,
            oy: y,
            rx: rng.gen(),
            ry: rng.gen(),
            x,
            y,
            // Velocity
            vx: rng.gen_range(-1.0..1.0) / 2000.0,
            vy: rng.gen_range(-1.0..1.0) / 2000.0,
            ocolor: c,
            color: (0.0, 0.0, 0.0),
            state: [false, false]
        };

        s.set_state(0, alive);

        return s;
    }
}

// Random nice looking rules
fn nice_rules() -> Vec<Rule> {
    return vec![
        vec![(0.0, false), (0.2, false), (0.4, true), (0.6000000000000001, true), (0.8, true), (1.0, true)],
        vec![(0.0, false), (0.01, false), (0.02, false), (0.03, false), (0.04, false), (0.05, false), (0.06, false), (0.07, false), (0.08, false), (0.09, false), (0.1, true), (0.11, true), (0.12, true), (0.13, true), (0.14, false), (0.15, false), (0.16, false), (0.17, false), (0.18, false), (0.19, false), (0.2, false), (0.21, false), (0.22, true), (0.23, true), (0.24, true), (0.25, true), (0.26, false), (0.27, false), (0.28, false), (0.29, false), (0.3, false), (0.31, false), (0.32, true), (0.33, true), (0.34, false), (0.35000000000000003, false), (0.36, false), (0.37, false), (0.38, false), (0.39, false), (0.4, false), (0.41000000000000003, false), (0.42, false), (0.43, false), (0.44, false), (0.45, false), (0.46, false), (0.47000000000000003, false), (0.48, false), (0.49, false), (0.5, false), (0.51, false), (0.52, false), (0.53, false), (0.54, true), (0.55, true), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, false), (0.61, false), (0.62, false), (0.63, false), (0.64, false), (0.65, false), (0.66, true), (0.67, true), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, true), (0.71, true), (0.72, false), (0.73, false), (0.74, false), (0.75, false), (0.76, false), (0.77, false), (0.78, false), (0.79, false), (0.8, false), (0.81, false), (0.8200000000000001, true), (0.8300000000000001, true), (0.84, false), (0.85, false), (0.86, true), (0.87, true), (0.88,true), (0.89, true), (0.9, false), (0.91, false), (0.92, false), (0.93, false), (0.9400000000000001, false), (0.9500000000000001, false), (0.96, false), (0.97, false), (0.98, false), (0.99, false)],
        vec![(0.0, false), (0.01, false), (0.02, true), (0.03, true), (0.04, false), (0.05, false), (0.06, false), (0.07, false), (0.08, false), (0.09, false), (0.1, false), (0.11, false), (0.12, false), (0.13, false), (0.14, false), (0.15, false), (0.16, true), (0.17, true), (0.18, false), (0.19, false), (0.2, false), (0.21, false), (0.22, false), (0.23, false), (0.24, false), (0.25, false), (0.26, false), (0.27, false), (0.28, false), (0.29, false), (0.3, false), (0.31, false), (0.32, false), (0.33, false), (0.34, false), (0.35000000000000003, false), (0.36, false), (0.37, false), (0.38, false), (0.39, false), (0.4, false), (0.41000000000000003, false), (0.42, false), (0.43, false), (0.44, true), (0.45, true), (0.46, false), (0.47000000000000003, false), (0.48, false), (0.49, false), (0.5, false), (0.51, false), (0.52, true), (0.53, true), (0.54,false), (0.55, false), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, true), (0.61, true), (0.62, false), (0.63, false), (0.64, true), (0.65, true), (0.66, false), (0.67, false), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, false), (0.71, false), (0.72, false), (0.73, false), (0.74, false), (0.75, false), (0.76, true), (0.77, true), (0.78, false), (0.79, false), (0.8, false), (0.81, false), (0.8200000000000001, false), (0.8300000000000001, false), (0.84, false), (0.85, false), (0.86, true), (0.87, true), (0.88, false), (0.89, false), (0.9, false), (0.91, false), (0.92, false), (0.93, false), (0.9400000000000001, true), (0.9500000000000001, true), (0.96, true), (0.97, true), (0.98, false), (0.99, false)],
        vec![(0.0, false), (0.01, false), (0.02, false), (0.03, false), (0.04, true), (0.05, true), (0.06, false), (0.07, false), (0.08, true), (0.09, true), (0.1, false), (0.11, false), (0.12, false), (0.13, false), (0.14, false), (0.15, false), (0.16, false), (0.17, false), (0.18, true), (0.19, true), (0.2, true), (0.21, true), (0.22, false), (0.23, false), (0.24, false), (0.25, false), (0.26, false), (0.27, false), (0.28, false), (0.29, false), (0.3, false), (0.31, false), (0.32, true), (0.33, true), (0.34, false), (0.35000000000000003, false), (0.36, false), (0.37, false), (0.38, false), (0.39, false), (0.4, true), (0.41000000000000003, true), (0.42, false), (0.43, false), (0.44, false), (0.45, false), (0.46, false), (0.47000000000000003, false), (0.48, false), (0.49, false), (0.5, true), (0.51, true), (0.52, true), (0.53, true), (0.54, false), (0.55, false), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, false), (0.61, false), (0.62, false), (0.63, false), (0.64, false), (0.65, false), (0.66, false), (0.67, false), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, false), (0.71, false), (0.72, false), (0.73, false), (0.74, false), (0.75, false), (0.76, true), (0.77, true), (0.78, false), (0.79, false), (0.8, false), (0.81, false), (0.8200000000000001, true), (0.8300000000000001, true), (0.84, true), (0.85, true), (0.86, false), (0.87, false), (0.88, false), (0.89, false), (0.9, false), (0.91, false), (0.92, false), (0.93, false), (0.9400000000000001, true), (0.9500000000000001, true), (0.96, true), (0.97, true), (0.98, true), (0.99, true)],
        vec![(0.0, true), (0.01, true), (0.02, false), (0.03, false), (0.04, true), (0.05, true), (0.06, false), (0.07, false), (0.08, true), (0.09, true), (0.1, true), (0.11, true), (0.12, false), (0.13, false), (0.14, false), (0.15, false), (0.16, false), (0.17, false), (0.18, true), (0.19, true), (0.2, false), (0.21, false), (0.22, false), (0.23, false), (0.24, false), (0.25, false), (0.26, false), (0.27, false), (0.28, false), (0.29, false), (0.3, true), (0.31, true), (0.32, false), (0.33, false), (0.34, false), (0.35000000000000003, false), (0.36, false), (0.37, false), (0.38, false), (0.39, false), (0.4, false), (0.41000000000000003, false), (0.42, false), (0.43, false), (0.44, false), (0.45, false), (0.46, false), (0.47000000000000003, false), (0.48, false), (0.49, false), (0.5, false), (0.51, false), (0.52, false), (0.53, false), (0.54, false), (0.55, false), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, false), (0.61, false), (0.62, true), (0.63, true), (0.64, false), (0.65, false), (0.66, false), (0.67, false), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, false), (0.71, false), (0.72, false), (0.73, false), (0.74, true), (0.75, true), (0.76, false), (0.77, false), (0.78, true), (0.79, true), (0.8, false), (0.81, false), (0.8200000000000001, false), (0.8300000000000001, false), (0.84, false), (0.85, false), (0.86, false), (0.87, false), (0.88, false), (0.89, false), (0.9, false), (0.91, false), (0.92, false), (0.93, false), (0.9400000000000001, false), (0.9500000000000001, false), (0.96, true), (0.97, true), (0.98, false), (0.99, false)],
        vec![(0.0, false), (0.01, false), (0.02, true), (0.03, true), (0.04, false), (0.05, false), (0.06, false), (0.07, false), (0.08, true), (0.09, true), (0.1, true), (0.11, true), (0.12, false), (0.13, false), (0.14, false), (0.15, false), (0.16, true), (0.17, true), (0.18, false), (0.19, false), (0.2, false), (0.21, false), (0.22, false), (0.23, false), (0.24, false), (0.25, false), (0.26, false), (0.27, false), (0.28, false), (0.29, false), (0.3, false), (0.31, false), (0.32, false), (0.33, false), (0.34, true), (0.35000000000000003, true), (0.36, false), (0.37, false), (0.38, false), (0.39, false), (0.4, false), (0.41000000000000003, false), (0.42, true), (0.43, true), (0.44, false), (0.45, false), (0.46, false), (0.47000000000000003, false), (0.48, true), (0.49, true), (0.5, false), (0.51, false), (0.52, false), (0.53, false), (0.54, true), (0.55, true), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, false), (0.61, false), (0.62, false), (0.63, false), (0.64, false), (0.65, false), (0.66, false), (0.67, false), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, false), (0.71, false), (0.72, false), (0.73, false), (0.74, false), (0.75, false), (0.76, false), (0.77, false), (0.78, true), (0.79, true), (0.8, false), (0.81, false), (0.8200000000000001, true), (0.8300000000000001, true), (0.84, false), (0.85, false), (0.86, false), (0.87, false), (0.88, false), (0.89, false), (0.9, false), (0.91, false), (0.92, true), (0.93, true), (0.9400000000000001, false), (0.9500000000000001, false), (0.96, false), (0.97, false), (0.98, false), (0.99, false)],
        vec![(0.0, false), (0.01, false), (0.02, false), (0.03, false), (0.04, false), (0.05, false), (0.06, false), (0.07, false), (0.08, true), (0.09, true), (0.1, true), (0.11, true), (0.12, false), (0.13, false), (0.14, false), (0.15, false), (0.16, true), (0.17, true), (0.18, false), (0.19, false), (0.2, false), (0.21, false), (0.22, false), (0.23, false), (0.24, false), (0.25, false), (0.26, false), (0.27, false), (0.28, false), (0.29, false), (0.3, false), (0.31, false), (0.32, true), (0.33, true), (0.34, true), (0.35000000000000003, true), (0.36, false), (0.37, false), (0.38, false), (0.39, false), (0.4, false), (0.41000000000000003, false), (0.42, false), (0.43, false), (0.44, false), (0.45, false), (0.46, true), (0.47000000000000003, true), (0.48, false), (0.49, false), (0.5, true), (0.51, true), (0.52, false), (0.53, false), (0.54, true), (0.55, true), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, false), (0.61, false), (0.62, false), (0.63, false), (0.64, false), (0.65, false), (0.66, false), (0.67, false), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, true), (0.71, true), (0.72, false), (0.73, false), (0.74, true), (0.75, true), (0.76, true), (0.77, true), (0.78, false), (0.79, false), (0.8, false), (0.81, false), (0.8200000000000001, false), (0.8300000000000001, false), (0.84, false), (0.85, false), (0.86, true), (0.87, true), (0.88, true), (0.89, true), (0.9, false), (0.91, false), (0.92, false), (0.93, false), (0.9400000000000001, false), (0.9500000000000001, false), (0.96, false), (0.97, false), (0.98, false), (0.99, false)],
        vec![(0.0, false), (0.01, false), (0.02, false), (0.03, false), (0.04, true), (0.05, true), (0.06, false), (0.07, false), (0.08, false), (0.09, false), (0.1, true), (0.11, true), (0.12, false), (0.13, false), (0.14, true), (0.15, true), (0.16, false), (0.17, false), (0.18, false), (0.19, false), (0.2, true), (0.21, true), (0.22, false), (0.23, false), (0.24, false), (0.25, false), (0.26, true), (0.27, true), (0.28, false), (0.29, false), (0.3, false), (0.31, false), (0.32, false), (0.33, false), (0.34, false), (0.35000000000000003, false), (0.36, true), (0.37, true), (0.38, false), (0.39, false), (0.4, false), (0.41000000000000003, false), (0.42, true), (0.43, true), (0.44, false), (0.45, false), (0.46, true), (0.47000000000000003, true), (0.48, true), (0.49, true), (0.5, true), (0.51, true), (0.52, true), (0.53, true), (0.54, false), (0.55, false), (0.56, false), (0.5700000000000001, false), (0.58, false), (0.59, false), (0.6, false), (0.61, false), (0.62, false), (0.63, false), (0.64, false), (0.65, false), (0.66, false), (0.67, false), (0.68, false), (0.6900000000000001, false), (0.7000000000000001, false), (0.71, false), (0.72, false), (0.73, false), (0.74, false), (0.75, false), (0.76, false), (0.77, false), (0.78, false), (0.79, false), (0.8, false), (0.81, false), (0.8200000000000001, true), (0.8300000000000001, true), (0.84, false), (0.85, false), (0.86, false), (0.87, false), (0.88, true), (0.89, true), (0.9, false), (0.91, false), (0.92, false), (0.93, false), (0.9400000000000001, false), (0.9500000000000001, false), (0.96, false), (0.97, false), (0.98, false), (0.99, false)]
    ];
}

pub struct CellularAutomaton {
    cur_voro: VoronoiDiagram,
    cells: Vec<Cell>,
    flipped: bool,
    do_animate: bool,
    rule: Rule,
    rule2: Rule2,
    use_rule2: bool,
    nice_rule_idx: usize,
    iter: usize,
    gol_hack: bool
}

impl CellularAutomaton {
    // Generate randomly placed initial points
    fn generate_points(count: usize) -> Vec<Cell> {
        let mut points = Vec::with_capacity(count);
        let mut rng = rand::thread_rng();
    
        // +4 = fake corner points
        for _ in 0..count + 4 {
            points.push(Cell::new(rng.gen(), rng.gen(), rng.gen()));
        }
    
        return points;
    }

    // Generate initial points in a grid
    fn generate_grid(count: usize) -> Vec<Cell> {
        let mut rng = rand::thread_rng();
        let mut points = Vec::new();
        let dim = 8;
        let incr = 0.1; // align nicely in viewport
        for x in -dim..=dim {
            for y in -dim..=dim {
                points.push(Cell::new(x as f64 * incr as f64, y as f64 * incr as f64, rng.gen()));
            }
        }
        return points;
    }

    fn random_rule() -> Rule {
        // (live_neighbor_ratio, new_state)
        let count = 100;
        let incr = 1.0 / count as f64;
        let mut rng = rand::thread_rng();
        let mut rule = Vec::with_capacity(count);
        for i in (0..count).step_by(2) {
            let state = rng.gen_bool(0.25);
            let min = i as f64 * incr;
            let max = (i + 1) as f64 * incr;
            rule.push((min, state));
            rule.push((max, state));
        }
        println!("randomizing rule");
        println!("{:?}", rule);
        return rule;
    }

    fn random_rule2() -> Rule2 {
        let count = 10;
        let incr = 1.0 / count as f64;
        let mut rng = rand::thread_rng();
        let mut b = Vec::with_capacity(count);
        let mut s = Vec::with_capacity(count);
        for i in 0..count {
            let bs = rng.gen_bool(0.1);
            let ss = rng.gen_bool(0.25);
            let val = i as f64 * incr;
            b.push((val, bs));
            s.push((val, ss));
        }
        let rule = [b, s];
        println!("randomizing rule2");
        println!("{:?}", rule);
        return rule;
    }

    pub fn randomize_rule(&mut self) {
        if self.use_rule2 {
            self.rule2 = CellularAutomaton::random_rule2();
        } else {
            self.rule = CellularAutomaton::random_rule();
        }
    }

    pub fn randomize_state(&mut self) {
        println!("randomizing state");
        let mut rng = rand::thread_rng();
        for cell in self.cells.iter_mut() {
            let s = rng.gen_bool(0.5);
            cell.set_state(0, s);
            cell.set_state(1, s);
        }
    }

    pub fn nice_rule(&mut self) {
        // Use next rule in list
        let nice = nice_rules();
        self.nice_rule_idx += 1;
        self.nice_rule_idx %= nice.len();
        self.rule = nice[self.nice_rule_idx].clone();
    }

    fn bounds() -> ((f64, f64), (f64, f64)) {
        // Voronoi clipping bounds
        let dim = 2.5;
        return ((-dim, -dim), (dim, dim));
    }

    fn new_voronoi(points: &[Cell]) -> VoronoiDiagram {

        let sites: Vec<(f64, f64)> = points.iter()
            .take(points.len() - 4) // Exclude corners
            .map(|p| (p.x, p.y))
            .collect();
        let (min, max) = CellularAutomaton::bounds();
        return voronator::VoronoiDiagram::<voronator::delaunator::Point>::from_tuple(&min, &max, &sites).unwrap();
    }
    
    pub fn new(point_count: usize) -> Self {
        let cells = CellularAutomaton::generate_grid(point_count);
        let cur_voro = CellularAutomaton::new_voronoi(&cells);
        //let rule = CellularAutomaton::random_rule();
        let nice_rule_idx = 0;
        let rule = nice_rules()[nice_rule_idx].clone();

        let rule2 = {
            // Game of life
            let incr = 1.0 / 8.0;
            let b = vec![false, false, false, true, false, false, false, false, false]
                .iter()
                .enumerate()
                .map(|(i, &s)| (i as f64 * incr, s))
                .collect();
            let s = vec![false, false, true,  true, false, false, false, false, false]
                .iter()
                .enumerate()
                .map(|(i, &s)| (i as f64 * incr, s))
                .collect();
            [b, s]
        };

        return Self {
            cur_voro,
            cells,
            flipped: false,
            do_animate: false,
            rule,
            rule2,
            use_rule2: true,
            nice_rule_idx,
            iter: 0,
            gol_hack: true
        };
    }

    fn animate(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap().as_millis() as f64;
            
        let n = self.cells.len() - 4; // Exclude corners
        for point in self.cells.iter_mut().take(n) {
            // Apply velocity
            point.x += point.vx;
            point.y += point.vy;

            // Invert when hits wall
            if point.x > 1.0 || point.x < -1.0 {
                point.vx *= -1.0;
            }

            if point.y > 1.0 || point.y < -1.0 {
                point.vy *= -1.0;
            }
        }
    }

    pub fn toggle_animation(&mut self) {
        self.do_animate = !self.do_animate;
    }

    pub fn create_vertices(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
    
        // Iterate over voronoi cells
        for (cell_idx, cell) in self.cur_voro.cells().iter().enumerate() {
            // Create vertices by triangulating cells
            let cell_points = cell.points();
    
            if let Some(diagram) = voronator::CentroidDiagram::<voronator::delaunator::Point>::new(cell_points) {
                // Get color from original site point
                let c = &self.cells[cell_idx].color;
        
                // Iterate over cell's triangles
                for tri_idx in diagram.delaunay.triangles {
                    let tri_p = &cell_points[tri_idx];
                    vertices.push(vertex(tri_p.x, tri_p.y,
                                         c.0, c.1, c.2));
                }
            }
        }
    
        indices.reserve_exact(vertices.len());
        for i in 0..vertices.len() {
            indices.push(i as u32);
        }
    
        return (vertices, indices);
    }

    fn apply_rule(&self, cell_state: bool, state_ratio: f64) -> bool {
        for range in self.rule.windows(2) {
            let (min, new_state) = range[0];
            let max = range[1].0;
            if state_ratio >= min && state_ratio <= max {
                return new_state;
            }
        }

        return false;
    }

    fn apply_rule2(&self, cell_state: bool, count: usize, state_ratio: f64) -> bool {
        let i = if cell_state { 1 } else { 0 };
        let mut rng = rand::thread_rng();
        
        let r = &self.rule2[i];
        if count < r.len() {
            let n = r[count].1;
            // A hack to help game of life not die so easily due to inaccurate neighbor detection
            if self.gol_hack && self.do_animate {
                if !n && rng.gen_bool(0.1) {
                    let a = (count - 1) % r.len();
                    let b = (count + 1) % r.len();
                    return r[a].1 || r[b].1;
                }
            }
            return n;
        } else {
            return false;
        }
    }

    fn step(&mut self) {
        let (read_idx, write_idx) = if self.flipped { (1, 0) } else { (0, 1) };
        self.flipped = !self.flipped;

        let moore = [
            (-1.0, -1.0), ( 0.0, -1.0), ( 1.0, -1.0),
            (-1.0,  0.0),               ( 1.0,  0.0),
            (-1.0,  1.0), ( 0.0,  1.0), ( 1.0,  1.0)
        ];

        // Iterate over voronoi cells
        for i in 0..self.cells.len() {
            // Count live neighbors
            let mut neigh_count = 0;
            let mut live_neighs = 0;

            if self.do_animate {
                // Try find neighbors from voronoi structure
                self.cur_voro.neighbors[i].iter().for_each(|neigh_idx| {
                    neigh_count += 1;
                    if self.cells[*neigh_idx].state[read_idx] {
                        live_neighs += 1;
                    }
                });
            } else {
                // Use a more accurate neighbor detection method while not animating
                let cur = self.cells[i];
                for (ox, oy) in moore {
                    let x = cur.x + ox / 10.0;
                    let y = cur.y + oy / 10.0;
                    for other in &self.cells {
                        if feq(other.x, x) && feq(other.y, y) {
                            neigh_count += 1;
                            if other.state[read_idx] {
                                live_neighs += 1;
                            }
                        }
                    }
                }
            }
            
            let ratio = live_neighs as f64 / neigh_count as f64;
            
            // Compute new cell state
            let old_state = self.cells[i].state[read_idx];
            let new_state = if self.use_rule2 {
                self.apply_rule2(old_state, live_neighs, ratio)
            } else {
                self.apply_rule(old_state, ratio)
            };
            self.cells[i].set_state(write_idx, new_state);
        }
    }

    pub fn update(&mut self) {
        self.iter += 1;
        let do_step = self.iter % 10 == 0;

        // Start demo
        if self.iter % 600 == 0 {
            self.do_animate = true;
        }

        if do_step {
            self.step();
        }

        if self.do_animate {
            if self.iter > 5000 {
                self.use_rule2 = false;
            }
            if self.iter > 1000 && self.iter % 300 == 0 {
                self.gol_hack = false;
                self.randomize_rule();
                self.randomize_state();
            }
            self.animate();
        }

        if self.do_animate || do_step {
            // Recompute voronoi when something changes
            self.cur_voro = CellularAutomaton::new_voronoi(&self.cells);
        }
    }

}
