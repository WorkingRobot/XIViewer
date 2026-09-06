//! TitleEdit presets, which stand a capture and this viewer in the same place.
//!
//! The plugin renders a real zone behind the title screen with no NPCs and no players in the frame,
//! and states everything about that frame in one file: which zone, where the camera stood, what it
//! looked at, the field of view, the weather and the hour. Reading one back puts this view where the
//! capture was taken from, which is what makes the two comparable.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::f32::consts::PI;

use base64::{Engine, prelude::BASE64_STANDARD};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Seconds in a day, which the offset wraps into.
const DAY: f32 = 86_400.0;

/// How far in front of the camera the point it looks at is written, which the plugin states as a
/// place in the world rather than as a direction. Anything along the ray says the same thing.
const TOWARD: f32 = 10.0;

/// The shape the plugin writes today, which it refuses to read a preset of any other version by.
const VERSION: u32 = 6;

/// One slot per festival the lobby can stand under, all of them empty here.
const FESTIVALS: usize = 8;

/// What the plugin puts in front of a preset handed over the clipboard, whose body is its JSON in
/// base64. It writes the first and reads either.
const MARKERS: [&str; 2] = ["TE3", "TE2"];

/// The lobby palette a preset carries, which the plugin defaults to Dawntrail's.
const DAWNTRAIL: UiColor = UiColor {
    expansion: 5,
    color: Rgba {
        x: 0.988_235_3,
        y: 0.862_745_1,
        z: 0.592_156_9,
        w: 1.0,
    },
    edge_color: Rgba {
        x: 0.298_039_23,
        y: 0.203_921_57,
        z: 0.184_313_73,
        w: 1.0,
    },
    highlight_color: Rgba {
        x: 1.0,
        y: 0.247_058_81,
        z: -0.121_568_62,
        w: 0.8,
    },
};

#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Rgba {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(default, rename_all = "PascalCase")]
struct UiColor {
    expansion: u32,
    color: Rgba,
    edge_color: Rgba,
    highlight_color: Rgba,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(default, rename_all = "PascalCase")]
struct Mount {
    last_location_mount: bool,
    mount_id: u32,
    buddy_model_top: u32,
    buddy_model_body: u32,
    buddy_model_legs: u32,
    buddy_stain: u32,
}

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "PascalCase")]
struct Festival {
    id: u16,
    phase: u16,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct File {
    version: u32,
    name: String,
    author: String,
    camera_follow_mode: u32,
    last_location_mount: bool,
    location_model: Location,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(default, rename_all = "PascalCase")]
struct Location {
    version: u32,
    location_type: u32,
    title_screen_logo: u32,
    territory_path: String,
    territory_type_id: u32,
    layout_territory_type_id: u32,
    layout_layer_filter_key: u32,
    position: Point,
    camera_position: Point,
    rotation: f32,
    yaw: f32,
    roll: f32,
    pitch: f32,
    fov: f32,
    weather_id: u8,
    time_offset: u16,
    bgm_id: u32,
    bgm_path: Option<String>,
    movement_mode: u32,
    mount: Mount,
    active: Vec<u64>,
    inactive: Vec<u64>,
    vfx_trigger_indexes: BTreeMap<u64, i16>,
    festivals: Vec<Festival>,
    save_layout: bool,
    use_vfx: bool,
    save_housing: bool,
    save_festivals: bool,
    title_screen_override: Option<u32>,
    ui_color: UiColor,
    title_screen_movie: i32,
    use_live_time: bool,
    furniture: Option<Value>,
    plots: Option<Value>,
    estate: Option<Value>,
}

/// The shape the plugin wrote before it kept a whole location, which it still reads.
#[derive(Deserialize)]
struct Older {
    #[serde(rename = "Name")]
    name: Option<String>,
    #[serde(rename = "TerritoryPath")]
    territory: String,
    #[serde(rename = "CameraPos")]
    camera: Point,
    #[serde(rename = "FixOnPos")]
    toward: Point,
    #[serde(rename = "FovY")]
    fov: Option<f32>,
    #[serde(rename = "WeatherId")]
    weather: Option<u32>,
    #[serde(rename = "TimeOffset")]
    time: Option<f32>,
}

/// One preset as this view uses it.
pub struct Preset {
    pub name: String,
    /// The level the plugin names, as the path this viewer opens it by.
    pub level: String,
    pub camera: Vec3,
    pub toward: Vec3,
    pub fov: Option<f32>,
    pub weather: Option<u32>,
    /// Seconds since midnight, where the file states an offset.
    pub time: Option<f32>,
}

impl Preset {
    pub fn read(bytes: &[u8]) -> Result<Self, String> {
        let bytes = unwrapped(bytes)?;
        match serde_json::from_slice::<Older>(&bytes) {
            Ok(held) => Ok(held.into()),
            Err(_) => serde_json::from_slice::<File>(&bytes)
                .map(Into::into)
                .map_err(|why| why.to_string()),
        }
    }

    /// Which way the camera looks, as the two angles this view holds one by.
    pub fn angles(&self) -> (f32, f32) {
        let held = (self.toward - self.camera).normalize_or_zero();
        (held.x.atan2(held.z), held.y.clamp(-1.0, 1.0).asin())
    }

    /// The view as the plugin would have written it, so a place found here can be stood in again in
    /// the game and captured.
    pub fn of(
        level: &str,
        camera: Vec3,
        forward: Vec3,
        fov: f32,
        weather: Option<u32>,
        time: f32,
    ) -> Self {
        let stem = level.rsplit('/').next().unwrap_or(level);
        Self {
            name: stem.trim_end_matches(".lvb").to_owned(),
            level: level.to_owned(),
            camera,
            toward: camera + forward.normalize_or_zero() * TOWARD,
            fov: Some(fov),
            weather,
            time: Some(time),
        }
    }

    pub fn write(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.file()).map_err(|why| why.to_string())
    }

    /// The preset as the plugin hands one over the clipboard.
    pub fn share(&self) -> Result<String, String> {
        let held = serde_json::to_string(&self.file()).map_err(|why| why.to_string())?;
        Ok(format!("{}{}", MARKERS[0], BASE64_STANDARD.encode(held)))
    }

    fn file(&self) -> File {
        let point = |held: Vec3| Point {
            x: held.x,
            y: held.y,
            z: held.z,
        };
        let (yaw, pitch) = self.angles();
        File {
            version: VERSION,
            name: self.name.clone(),
            author: String::new(),
            camera_follow_mode: 0,
            last_location_mount: false,
            location_model: Location {
                version: VERSION,
                territory_path: self
                    .level
                    .trim_start_matches("bg/")
                    .trim_end_matches(".lvb")
                    .to_owned(),
                position: point(self.camera),
                camera_position: point(self.camera),
                yaw,
                pitch,
                fov: self.fov.map_or(1.0, f32::to_radians),
                weather_id: self.weather.unwrap_or_default() as u8,
                time_offset: self.time.map_or(0, offset),
                festivals: vec![Festival::default(); FESTIVALS],
                use_vfx: true,
                save_housing: true,
                ui_color: DAWNTRAIL,
                title_screen_movie: -1,
                ..Default::default()
            },
        }
    }
}

impl From<File> for Preset {
    fn from(
        File {
            name,
            location_model: held,
            ..
        }: File,
    ) -> Self {
        let camera = Vec3::new(
            held.camera_position.x,
            held.camera_position.y,
            held.camera_position.z,
        );
        let (yaw, pitch) = (held.yaw, held.pitch);
        let toward = Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        );
        Self {
            name: match name.is_empty() {
                true => stem(&held.territory_path).to_owned(),
                false => name,
            },
            level: format!("bg/{}.lvb", held.territory_path),
            camera,
            toward: camera + toward * TOWARD,
            // The plugin holds this in radians and its own slider stops at three, so a wider angle
            // is one the older shape carried in degrees and its migration copied over untouched.
            fov: Some(match held.fov > PI {
                true => held.fov,
                false => held.fov.to_degrees(),
            }),
            weather: Some(held.weather_id.into()),
            time: Some(clock(held.time_offset.into())),
        }
    }
}

impl From<Older> for Preset {
    fn from(held: Older) -> Self {
        Self {
            name: held
                .name
                .unwrap_or_else(|| stem(&held.territory).to_owned()),
            // The plugin states the path without its extension, and the stem twice over: the
            // directory the level sits in and the file itself go by the same name.
            level: format!("bg/{}.lvb", held.territory),
            camera: Vec3::new(held.camera.x, held.camera.y, held.camera.z),
            toward: Vec3::new(held.toward.x, held.toward.y, held.toward.z),
            // The plugin holds this in radians in every shape it has ever written, `File` included:
            // see the same conversion there.
            fov: held.fov.map(|fov| match fov > PI {
                true => fov,
                false => fov.to_degrees(),
            }),
            weather: held.weather,
            time: held.time.map(|held| clock(held as u32)),
        }
    }
}

fn stem(territory: &str) -> &str {
    territory.rsplit('/').next().unwrap_or(territory)
}

/// The hour an offset states, in seconds since midnight. It reads as `hhmm` with the minutes
/// allowed to run past sixty: 240 is 02:40, 640 is 06:40 and 1985 is 19:85, which is 20:25.
fn clock(held: u32) -> f32 {
    ((held / 100 * 3600 + held % 100 * 60) as f32).rem_euclid(DAY)
}

/// The offset an hour is written as.
fn offset(held: f32) -> u16 {
    let minutes = (held / 60.0).round() as u32;
    (minutes / 60 * 100 + minutes % 60) as u16
}

/// The JSON a marked preset carries, or the bytes themselves where they carry no marker.
/// Whether a paste is worth reading as a preset at all, so one landing in a text field somewhere
/// else is left alone rather than reported as a broken one.
pub fn looks_like(text: &str) -> bool {
    let held = text.trim();
    held.starts_with('{') || MARKERS.iter().any(|marker| held.starts_with(marker))
}

fn unwrapped(bytes: &[u8]) -> Result<Cow<'_, [u8]>, String> {
    let held = str::from_utf8(bytes)
        .map(str::trim)
        .ok()
        .and_then(|held| MARKERS.iter().find_map(|marker| held.strip_prefix(marker)));
    let Some(held) = held else {
        return Ok(bytes.into());
    };
    // Braces around the body are none of the plugin's doing, but a paste can carry them.
    let held = held
        .strip_prefix('{')
        .and_then(|held| held.strip_suffix('}'))
        .unwrap_or(held);
    BASE64_STANDARD
        .decode(held)
        .map(Cow::Owned)
        .map_err(|why| why.to_string())
}

thread_local! {
    /// A preset waiting for the level it names to open. Opening one builds a scene of its own, so
    /// there is nowhere inside a scene for it to live across that.
    static PENDING: RefCell<Option<Preset>> = const { RefCell::new(None) };
}

/// Keeps a preset until the level it names has opened.
pub fn hold(held: Preset) {
    PENDING.with(|slot| *slot.borrow_mut() = Some(held));
}

/// The preset a scene was opened for, where it was opened for one. Left in place for a scene
/// opened for somewhere else, since that scene is not the one it was held for.
pub fn taken(level: &str) -> Option<Preset> {
    PENDING.with(|slot| {
        let matches = slot.borrow().as_ref().is_some_and(|held| held.level == level);
        matches.then(|| slot.borrow_mut().take()).flatten()
    })
}

#[cfg(test)]
mod test {
    use glam::Vec3;

    use super::Preset;

    /// The preset the user captured Ishgard from, before the plugin kept a whole location.
    #[test]
    fn a_preset_reads_as_the_plugin_once_wrote_it() {
        let held = Preset::read(
            br#"{
                "Name": "TE_Ishgard",
                "TerritoryPath": "ex1/01_roc_r2/twn/r2t1/level/r2t1",
                "CameraPos": { "X": -251.920822, "Y": 8.874063, "Z": 166.831223 },
                "FixOnPos": { "X": -245.1769, "Y": 12.92161, "Z": 160.655716 },
                "FovY": 45.0,
                "WeatherId": 15,
                "TimeOffset": 1985
            }"#,
        )
        .expect("a preset");
        assert_eq!(held.level, "bg/ex1/01_roc_r2/twn/r2t1/level/r2t1.lvb");
        assert_eq!(held.weather, Some(15));
        // `hhmm` with the minutes running over: 1985 is 19:85, which is 20:25.
        assert_eq!(held.time, Some(20.0 * 3600.0 + 25.0 * 60.0));
        let (yaw, pitch) = held.angles();
        // It looks back toward positive x and slightly up, which is what the two points say.
        assert!(yaw > 0.0 && pitch > 0.0);
        assert!((pitch.to_degrees() - 23.0).abs() < 1.0);
    }

    /// The same preset once the plugin had converted it, which states the two angles rather than a
    /// point to look at and carries the field of view its own migration left in degrees.
    #[test]
    fn a_preset_reads_as_the_plugin_writes_it_now() {
        let held = Preset::read(
            br#"{"Version":6,"Name":"TE_Ishgard2","Author":"","CameraFollowMode":0,
            "LastLocationMount":false,"LocationModel":{"Version":6,"LocationType":0,
            "TitleScreenLogo":0,"TerritoryPath":"ex1/01_roc_r2/twn/r2t1/level/r2t1",
            "TerritoryTypeId":418,"LayoutTerritoryTypeId":0,"LayoutLayerFilterKey":0,
            "Position":{"X":-251.92082,"Y":8.874063,"Z":166.83122},
            "CameraPosition":{"X":-251.92082,"Y":8.874063,"Z":166.83122},"Rotation":0.0,
            "Yaw":2.3122256,"Roll":0.0,"Pitch":0.41671044,"Fov":45.0,"WeatherId":15,
            "TimeOffset":240,"BgmId":318,"BgmPath":"music/ex1/BGM_EX1_Event_Start.scd",
            "MovementMode":0,"Mount":{"LastLocationMount":false,"MountId":0,"BuddyModelTop":0,
            "BuddyModelBody":0,"BuddyModelLegs":0,"BuddyStain":0},"Active":[],"Inactive":[],
            "VfxTriggerIndexes":{},"Festivals":[{"Id":0,"Phase":0}],"SaveLayout":false,
            "UseVfx":true,"SaveHousing":true,"SaveFestivals":false,"TitleScreenOverride":null,
            "TitleScreenMovie":-1,"UseLiveTime":false,"Furniture":null,"Plots":null,
            "Estate":null}}"#,
        )
        .expect("a preset");
        assert_eq!(held.name, "TE_Ishgard2");
        assert_eq!(held.level, "bg/ex1/01_roc_r2/twn/r2t1/level/r2t1.lvb");
        assert_eq!(held.weather, Some(15));
        assert_eq!(held.time, Some(2.0 * 3600.0 + 40.0 * 60.0));
        assert_eq!(held.fov, Some(45.0));
        let (yaw, pitch) = held.angles();
        assert!((yaw - 2.312_225_6).abs() < 1e-4 && (pitch - 0.416_710_44).abs() < 1e-4);
    }

    /// One written out and read back is the same view, so a place found here can be stood in again
    /// in the game and captured from.
    #[test]
    fn a_view_written_out_comes_back_the_same_view() {
        let held = Preset::of(
            "bg/ex1/01_roc_r2/twn/r2t1/level/r2t1.lvb",
            Vec3::new(-251.9, 8.9, 166.8),
            Vec3::new(0.6, 0.4, -0.7),
            45.0,
            Some(15),
            9.0 * 3600.0 + 5.0 * 60.0,
        );
        for text in [
            held.write().expect("written"),
            held.share().expect("shared"),
        ] {
            let back = Preset::read(text.as_bytes()).expect("read back");
            assert_eq!(back.level, held.level);
            assert_eq!(back.name, held.name);
            assert_eq!(back.weather, Some(15));
            assert!((back.fov.unwrap() - 45.0).abs() < 1e-3);
            assert!((back.time.unwrap() - held.time.unwrap()).abs() < 1.0);
            let (yaw, pitch) = back.angles();
            let (want_yaw, want_pitch) = held.angles();
            assert!((yaw - want_yaw).abs() < 1e-4 && (pitch - want_pitch).abs() < 1e-4);
        }
    }

    /// A preset pasted in with the marker the plugin hands one over the clipboard by, braces and
    /// all, since a paste can carry them.
    #[test]
    fn a_marked_preset_reads_out_of_a_paste() {
        let held = Preset::of(
            "bg/ex1/01_roc_r2/twn/r2t1/level/r2t1.lvb",
            Vec3::new(-251.9, 8.9, 166.8),
            Vec3::new(0.6, 0.4, -0.7),
            45.0,
            Some(15),
            9.0 * 3600.0 + 5.0 * 60.0,
        );
        let shared = held.share().expect("shared");
        let braced = format!("TE3{{{}}}", shared.trim_start_matches("TE3"));
        for text in [shared, braced] {
            assert_eq!(
                Preset::read(text.as_bytes()).expect("read back").level,
                held.level
            );
        }
    }
}
