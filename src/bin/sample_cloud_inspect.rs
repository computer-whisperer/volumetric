use std::path::PathBuf;

use volumetric::sample_cloud::SampleCloudDump;

fn main() {
    let mut path: Option<PathBuf> = None;
    let mut index: usize = 0;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--file" => {
                if let Some(value) = args.next() {
                    path = Some(PathBuf::from(value));
                }
            }
            "--set" => {
                if let Some(value) = args.next() {
                    index = value.parse::<usize>().unwrap_or(0);
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {}
        }
    }

    let Some(path) = path else {
        print_help();
        return;
    };

    let dump = match SampleCloudDump::load(&path) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("Failed to load: {err}");
            return;
        }
    };

    let Some(set) = dump.sets.get(index) else {
        eprintln!("Set {} not found (total {})", index, dump.sets.len());
        return;
    };

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for point in &set.points {
        for i in 0..3 {
            min[i] = min[i].min(point.position[i]);
            max[i] = max[i].max(point.position[i]);
        }
    }

    println!("Set {} label={:?}", index, set.label);
    println!("Vertex: {:?}", set.vertex);
    println!("Hint: {:?}", set.hint_normal);
    if let Some(bounds) = &set.cell_bounds {
        println!("Cell bounds: min={:?} max={:?}", bounds.min, bounds.max);
    }
    println!("Points: {}", set.points.len());
    println!("Sample min: {:?}", min);
    println!("Sample max: {:?}", max);
}

fn print_help() {
    eprintln!("Usage: sample_cloud_inspect --file <file.cbor> [--set <index>]");
}
