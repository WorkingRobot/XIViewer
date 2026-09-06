//! One set of a zone's environment timeline, keyframe by keyframe, and the pair straddling a given
//! time interpolated the way the engine reads it.
//!
//! `envs_set <zone> <weather> <seconds> <set kind>`

use ironworks::file::envs::Value;
use ironworks::file::{envb, lvb};
use ironworks::{
    Ironworks,
    sqpack::{Install, SqPack},
};

const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

fn shown(value: &Value) -> String {
    match value {
        Value::Float(held) => format!("{held:.6}"),
        Value::Unsigned(held) => held.to_string(),
        Value::Flag(held) => held.to_string(),
        Value::Colour(held) => format!(
            "{} {} {} a{} x{:.3}",
            held.red(),
            held.green(),
            held.blue(),
            held.alpha(),
            held.intensity()
        ),
        Value::Path(held) => held.clone(),
    }
}

fn main() {
    let ironworks = Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
    let mut args = std::env::args().skip(1);
    let zone = args.next().expect("zone");
    let wanted: u32 = args.next().and_then(|held| held.parse().ok()).unwrap_or(1);
    let at: f32 = args
        .next()
        .and_then(|held| held.parse().ok())
        .unwrap_or(0.0);
    let kind: u32 = args.next().and_then(|held| held.parse().ok()).unwrap_or(0);
    let stem = zone.rsplit('/').next().unwrap_or(&zone).to_owned();
    let path = match zone.starts_with("ffxiv/") || zone.starts_with("ex") {
        true => format!("bg/{zone}/level/{stem}.lvb"),
        false => format!("bg/ffxiv/{zone}/level/{stem}.lvb"),
    };
    // An `.envb` named outright rather than a zone, which is how a volume's own local environment
    // is reached: nothing states those in a level's own list.
    let paths: Vec<String> = match zone.ends_with(".envb") {
        true => vec![zone.clone()],
        false => {
            let level: lvb::LevelFile = ironworks.file(&path).unwrap();
            let mut paths: Vec<String> = Vec::new();
            for env in level.scene().environments() {
                let held = env.asset_path();
                if !held.is_empty() && !paths.contains(held) {
                    paths.push(held.clone());
                }
            }
            paths
        }
    };
    for held in paths {
        let file: envb::EnvironmentFile = ironworks.file(&held).unwrap();
        for weather in file.environments().weathers() {
            if weather.id() != wanted {
                continue;
            }
            for set in weather.sets().iter().filter(|set| set.kind() == kind) {
                println!("== {held} weather {wanted} set {kind} {:?}", set.name());
                let frames: Vec<_> = set.keyframes().iter().collect();
                for keyframe in &frames {
                    let fields: Vec<String> = keyframe
                        .fields()
                        .iter()
                        .map(|(name, value)| format!("{name}={}", shown(value)))
                        .collect();
                    println!("   {:>7.0}  {}", keyframe.time(), fields.join("  "));
                }
                let Some(two) = frames
                    .windows(2)
                    .find(|two| two[0].time() <= at && at <= two[1].time())
                else {
                    continue;
                };
                let u = (at - two[0].time()) / (two[1].time() - two[0].time());
                println!(
                    "\n   at {at} : u = {u:.6} between {} and {}",
                    two[0].time(),
                    two[1].time()
                );
                for (name, value) in two[0].fields() {
                    let after = two[1].fields().iter().find(|(other, _)| other == name);
                    let mix = |a: f32, b: f32| a + (b - a) * u;
                    match (value, after.map(|(_, held)| held)) {
                        (Value::Float(low), Some(Value::Float(high))) => {
                            println!("      {name} = {:.9}", mix(*low, *high));
                        }
                        (Value::Colour(low), Some(Value::Colour(high))) => {
                            let byte = |a: u8, b: u8| mix(f32::from(a), f32::from(b));
                            println!(
                                "      {name} = {:.6} {:.6} {:.6} a{:.6} x{:.6}   /255 = {:.9} {:.9} {:.9}",
                                byte(low.red(), high.red()),
                                byte(low.green(), high.green()),
                                byte(low.blue(), high.blue()),
                                byte(low.alpha(), high.alpha()),
                                mix(low.intensity(), high.intensity()),
                                byte(low.red(), high.red()) / 255.0,
                                byte(low.green(), high.green()) / 255.0,
                                byte(low.blue(), high.blue()) / 255.0,
                            );
                        }
                        _ => println!("      {name} = {}", shown(value)),
                    }
                }
            }
        }
    }
}
