//! Measures a viewer frame against a captured game frame.
//!
//!   framediff <game.png> [viewer.png] [--capture=x.zip.xml] [--state=state.json]
//!             [--crop=x0,y0,x1,y1] [--mask=...] [--grid=6x4] [--commit=<sha>] [--out=<dir>]
//!
//! With one image it reports that frame alone, which is how a recorded measurement is checked. With
//! two it stands them on one grid from the projections each was drawn under and reports the pair.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use framediff::capture::Capture;
use framediff::state::State;
use framediff::{
    Aligned, Cell, Rect, Region, Residual, Stats, View, align, difference, grid, overlay, worst,
};
use image::RgbImage;

fn main() -> ExitCode {
    match run() {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(why) => {
            eprintln!("framediff: {why}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    paths: Vec<PathBuf>,
    capture: Option<PathBuf>,
    state: Option<PathBuf>,
    crop: Option<Rect>,
    masks: Vec<Rect>,
    grid: (u32, u32),
    commit: Option<String>,
    out: Option<PathBuf>,
}

fn args() -> Result<Args, String> {
    let mut held = Args {
        paths: Vec::new(),
        capture: None,
        state: None,
        crop: None,
        masks: Vec::new(),
        grid: (6, 4),
        commit: None,
        out: None,
    };
    for one in std::env::args().skip(1) {
        let Some((name, value)) = one.strip_prefix("--").and_then(|one| one.split_once('=')) else {
            held.paths.push(PathBuf::from(one));
            continue;
        };
        match name {
            "capture" => held.capture = Some(PathBuf::from(value)),
            "state" => held.state = Some(PathBuf::from(value)),
            "crop" => held.crop = Some(Rect::read(value)?),
            "mask" => held.masks.push(Rect::read(value)?),
            "commit" => held.commit = Some(value.to_owned()),
            "out" => held.out = Some(PathBuf::from(value)),
            "grid" => {
                let (across, down) = value.split_once('x').ok_or("--grid wants <across>x<down>")?;
                held.grid = (
                    across.parse().map_err(|_| "--grid wants numbers")?,
                    down.parse().map_err(|_| "--grid wants numbers")?,
                );
            }
            _ => return Err(format!("--{name}: no such option")),
        }
    }
    match held.paths.is_empty() {
        true => Err("wants a frame to measure".to_owned()),
        false => Ok(held),
    }
}

fn run() -> Result<String, String> {
    let args = args()?;
    let game = framediff::open(&args.paths[0])?;
    let mut out = String::new();

    let capture = args
        .capture
        .as_deref()
        .map(Capture::open)
        .transpose()?
        .map(|held| -> Result<_, String> {
            let cameras = held.cameras()?;
            Ok((held, cameras))
        })
        .transpose()?;
    let state = args
        .state
        .as_deref()
        .map(|path| {
            fs::read_to_string(path)
                .map_err(|why| format!("{}: {why}", path.display()))
                .and_then(|held| {
                    serde_json::from_str::<State>(&held)
                        .map_err(|why| format!("{}: {why}", path.display()))
                })
        })
        .transpose()?;

    if let Some((held, cameras)) = &capture {
        let _ = writeln!(
            out,
            "Game    {}  thumbnail {}x{} of a {}x{} swapchain",
            held.name, held.thumbnail.0, held.thumbnail.1, held.extent.0, held.extent.1
        );
        match cameras.first() {
            Some(camera) => {
                let _ = writeln!(
                    out,
                    "        eye ({:.3}, {:.3}, {:.3})  looking ({:.3}, {:.3}, {:.3})  \
                     fov {:.2} vertical  near {}  {} copies, {} camera(s) stated",
                    camera.eye.x,
                    camera.eye.y,
                    camera.eye.z,
                    camera.forward.x,
                    camera.forward.y,
                    camera.forward.z,
                    camera.fov,
                    camera.near,
                    camera.copies,
                    cameras.len(),
                );
            }
            None => out.push_str("        no camera found in the frame's own writes\n"),
        }
    }
    if let Some(state) = &state {
        let _ = writeln!(
            out,
            "Viewer  {}{} built {}  {}{}",
            state.commit,
            match state.clean {
                true => "",
                false => " (dirty)",
            },
            state.built,
            state.level,
            state
                .preset
                .as_deref()
                .map_or_else(String::new, |held| format!("  preset {held}")),
        );
        let _ = writeln!(
            out,
            "        eye ({:.3}, {:.3}, {:.3})  looking ({:.3}, {:.3}, {:.3})  \
             fov {:.2} vertical  viewport {:.0}x{:.0} at ({:.0}, {:.0})",
            state.eye[0],
            state.eye[1],
            state.eye[2],
            state.forward[0],
            state.forward[1],
            state.forward[2],
            state.fov,
            state.viewport[2],
            state.viewport[3],
            state.viewport[0],
            state.viewport[1],
        );
        let _ = writeln!(
            out,
            "        {} weather {}  exposure {:.3} from a frame measuring {:.3}, step {:.3} s",
            state.clock(),
            state.weather,
            state.exposure,
            state.measured,
            state.step,
        );
        let _ = writeln!(
            out,
            "        drawn {} of {} placed, models {}, materials {}, lights {}, passes {}",
            state.drawn, state.placed, state.models, state.materials, state.lights, state.passes,
        );
        if let Some(want) = &args.commit
            && !want.starts_with(&state.commit)
            && !state.commit.starts_with(want.as_str())
        {
            return Err(format!(
                "the frame was drawn by {} and the tree is {want}: rebuild before measuring",
                state.commit
            ));
        }
    }
    if !out.is_empty() {
        out.push('\n');
    }

    let Some(view) = args.paths.get(1) else {
        let region = Region::new(args.crop.unwrap_or_else(|| Rect::of(&game)), args.masks);
        let _ = writeln!(out, "{}", place(&region));
        let _ = writeln!(out, "  {}", Stats::of(&game, &region).row());
        return Ok(out);
    };
    let view = framediff::open(view)?;

    let (view, region, native) = match (&capture, &state) {
        (Some((_, cameras)), Some(state)) => {
            let camera = cameras
                .first()
                .ok_or("the capture states no camera to stand the viewer against")?;
            let rect = state.viewport;
            let cut = Rect {
                x0: rect[0].round() as u32,
                y0: rect[1].round() as u32,
                x1: (rect[0] + rect[2]).round() as u32,
                y1: (rect[1] + rect[3]).round() as u32,
            };
            let inside = crop(&view, cut)?;
            let seen = camera.view(game.width(), game.height());
            let mine = state.view();
            let held = align(&inside, &mine, &seen);
            let _ = writeln!(
                out,
                "Alignment  the viewer's own frame is {}x{}, resampled onto the game's {}x{}",
                inside.width(),
                inside.height(),
                game.width(),
                game.height(),
            );
            let _ = write!(out, "{}", coverage(&held, &seen, &mine));
            let mut region =
                Region::new(args.crop.unwrap_or_else(|| Rect::of(&game)), args.masks.clone());
            let mut lost = 0;
            for y in region.rect.y0..region.rect.y1 {
                for x in region.rect.x0..region.rect.x1 {
                    if held.outside[(y * game.width() + x) as usize] && region.holds(x, y) {
                        region.drop(x, y);
                        lost += 1;
                    }
                }
            }
            let residual = Residual::between(&seen, &mine);
            let _ = writeln!(
                out,
                "           step {:.3} units  turn {:.3} deg  roll {:.3} deg  fov {:+.3} deg  \
                 parallax {:.3} deg at a hundred units, the turn worth {:.1} px here",
                residual.step,
                residual.turn,
                residual.roll,
                residual.fov,
                residual.parallax,
                residual.pixels,
            );
            let _ = writeln!(
                out,
                "           {lost} of the region's pixels lay past the viewer's own frame",
            );
            // Everything geometric here is read out of the capture. The hour and the weather are
            // not: they are whatever was handed to `rdframe`, and a wrong one makes every number
            // below a difference in environment rather than in shading.
            let _ = writeln!(
                out,
                "           camera, lens and frame shape are read out of the capture; the clock and \
                 the weather are stated\n",
            );
            let native = Stats::of(&inside, &Region::new(Rect::of(&inside), Vec::new()));
            (held.image, region, Some(native))
        }
        _ => {
            if view.dimensions() != game.dimensions() {
                return Err(format!(
                    "{}x{} against {}x{}: pass --capture and --state, or crop them to one shape",
                    game.width(),
                    game.height(),
                    view.width(),
                    view.height()
                ));
            }
            let region = Region::new(args.crop.unwrap_or_else(|| Rect::of(&game)), args.masks);
            (view, region, None)
        }
    };
    let _ = writeln!(out, "{}", place(&region));
    // A zone still streaming is the loudest confounder there is: what has not arrived reads as sky,
    // and the auto-exposure measures the frame it is in.
    if let Some(state) = &state
        && state.drawn * 10 < state.placed * 9
    {
        let _ = writeln!(
            out,
            "        the zone is {:.0}% placed and {} modelled, so what is missing reads as sky and \
             the exposure follows it",
            100.0 * state.drawn as f64 / state.placed.max(1) as f64,
            state.models,
        );
    }
    let (left, right) = (Stats::of(&game, &region), Stats::of(&view, &region));
    let _ = writeln!(out, "  game    {}", left.row());
    let _ = writeln!(out, "  viewer  {}", right.row());
    // Resampling onto a larger grid smooths, so what the viewer clips is only stated by its own
    // pixels. A clip or a percentile taken off the aligned row is not comparable with a recorded
    // one; the mean and the saturation barely move either way.
    if let Some(native) = &native {
        let _ = writeln!(out, "  its own {}   over its whole viewport", native.row());
    }
    let _ = writeln!(
        out,
        "  apart   gain {:.3}  saturation {:+.4}  per channel ({:.3}, {:.3}, {:.3})",
        left.luminance / right.luminance.max(f64::EPSILON),
        right.saturation - left.saturation,
        left.rgb[0] / right.rgb[0].max(f64::EPSILON),
        left.rgb[1] / right.rgb[1].max(f64::EPSILON),
        left.rgb[2] / right.rgb[2].max(f64::EPSILON),
    );

    let cells = grid(&game, &view, &region, args.grid.0, args.grid.1);
    let _ = writeln!(out, "\nGain, game over viewer, {} across", args.grid.0);
    let _ = write!(out, "{}", table(&cells, |cell| format!("{:6.2}", cell.gain())));
    let _ = writeln!(out, "\nSaturation, viewer less game");
    let _ = write!(out, "{}", table(&cells, |cell| format!("{:+6.3}", cell.tint())));
    let _ = writeln!(out, "\nRows, top to bottom");
    let rows = grid(&game, &view, &region, 1, args.grid.1 * 2);
    for row in &rows {
        for cell in row {
            let _ = writeln!(
                out,
                "  y {:4}-{:4}  gain {:5.2}  sat {:+.3}  lum {:6.2} against {:6.2}  \
                 clip {:5.2}% against {:5.2}%  black {:5.2}% against {:5.2}%",
                cell.rect.y0,
                cell.rect.y1,
                cell.gain(),
                cell.tint(),
                cell.game.luminance,
                cell.view.luminance,
                cell.game.clipped,
                cell.view.clipped,
                cell.game.black,
                cell.view.black,
            );
        }
    }
    let _ = writeln!(out, "\nWorst cells");
    let _ = write!(out, "{}", worst(&cells, 6));

    if let Some(dir) = &args.out {
        fs::create_dir_all(dir).map_err(|why| format!("{}: {why}", dir.display()))?;
        let write = |name: &str, held: &RgbImage| -> Result<(), String> {
            held.save(dir.join(name))
                .map_err(|why| format!("{name}: {why}"))
        };
        write("game.png", &game)?;
        write("view-aligned.png", &view)?;
        write("difference.png", &difference(&game, &view, &region, 4.0))?;
        write("overlay.png", &overlay(&game, &view, &region))?;
        fs::write(dir.join("report.txt"), &out).map_err(|why| why.to_string())?;
        let _ = writeln!(out, "\nwritten to {}", dir.display());
    }
    Ok(out)
}

fn place(region: &Region) -> String {
    format!(
        "Region  ({},{})-({},{}), {} pixels measured",
        region.rect.x0,
        region.rect.y0,
        region.rect.x1,
        region.rect.y1,
        region.pixels(),
    )
}

fn crop(image: &RgbImage, rect: Rect) -> Result<RgbImage, String> {
    if rect.x1 > image.width() || rect.y1 > image.height() || rect.width() == 0 {
        return Err(format!(
            "the viewer states a viewport of ({},{})-({},{}) and its frame is {}x{}",
            rect.x0,
            rect.y0,
            rect.x1,
            rect.y1,
            image.width(),
            image.height()
        ));
    }
    Ok(image::imageops::crop_imm(image, rect.x0, rect.y0, rect.width(), rect.height()).to_image())
}

/// How much of each frame the other one saw, which is what a difference in lens costs.
fn coverage(held: &Aligned, seen: &View, mine: &View) -> String {
    let reached = held.outside.iter().filter(|held| !**held).count();
    let whole = held.outside.len();
    format!(
        "           the viewer's frame covers {:.1}% of the game's, {:.1} deg wide against {:.1}\n",
        100.0 * reached as f64 / whole as f64,
        2.0 * (mine.aspect * (mine.fov.to_radians() * 0.5).tan())
            .atan()
            .to_degrees(),
        2.0 * (seen.aspect * (seen.fov.to_radians() * 0.5).tan())
            .atan()
            .to_degrees(),
    )
}

fn table(cells: &[Vec<Cell>], show: impl Fn(&Cell) -> String) -> String {
    let mut out = String::new();
    for row in cells {
        out.push(' ');
        for cell in row {
            let _ = write!(
                out,
                " {}",
                match cell.game.pixels {
                    0 => "     -".to_owned(),
                    _ => show(cell),
                }
            );
        }
        out.push('\n');
    }
    out
}
