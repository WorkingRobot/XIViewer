//! The light a zone states for itself, and controls for what it leaves to the engine.
//!
//! A scene names an environment per part of itself: an `.envb` holding a timeline per weather, and
//! the `EnvLocation` instance the part is centered on, which in turn names the `.amb` holding the
//! ambient light as spherical harmonics over the day. Both are read at one time of day and go into
//! the ambient buffer the composite reads.
//!
//! Everything neither file states is a control rather than a number invented here: which sky the
//! frame stands under, where the key light comes from, and every field of the ambient entry past the
//! harmonics and their scale.

use std::io::Cursor;
use std::ops::RangeInclusive;

use anyhow::Result;
use egui::RichText;
use glam::{Mat4, Vec2, Vec3, Vec4};
use ironworks::file::amb::{self, Ambient as AmbientFile, TRACK_COUNT};
use ironworks::file::envs::{Keyframe, Value};
use ironworks::file::{File, envb, layer};

use super::super::super::mdl::program;
use super::super::super::{link, section};
use crate::assets::deps::Deps;
use crate::backend::Backend;
use crate::utils::TrackedPromise;

/// Seconds in a day.
const DAY: f32 = 86_400.0;

/// Every sky the game holds, which its own id indexes.
const SKY_LIGHT: &str = "bgcommon/nature/sky/ambient/skylight.amb";

/// What a weather an `.envb` states a timeline for is called.
const WEATHER: &str = "Weather";

/// The sets of an `.envb` weather this reads: the light the whole zone is lit by, what the frame's
/// exposure and tone curve are worked out from, and the fog, which the file calls vertical.
const GLOBAL_LIGHTING: u32 = 0;
const CLOUDS: u32 = 2;
const STARFIELD: u32 = 12;
const WIND: u32 = 6;
const LIGHT_SHAFT: u32 = 7;
const WETNESS: u32 = 8;
const TONE_MAPPING: u32 = 9;
const VERTICAL_FOG: u32 = 13;

/// How far up the frame the moon's disc reaches, as a fraction of its height. A frame's own field of
/// view is not divided back out of it, so a wider one draws the disc small.
const MOON: f32 = 0.050_346;

/// Where the day panel opens: full, the same phase the moon's disc used to hold with no day of
/// its own to read.
const FULL: f32 = 17.0;

/// What the disc's own alpha falls off by where a weather states no starfield set.
const MOON_FADE: f32 = 0.4;

/// The slot of the level file's general block holding how far the sun's shadows reach.
const SHADOW_REACH: usize = 9;

/// What the two fog rates are stated per, rather than per unit of distance, and what the near
/// haze's own two are: twenty units of height and a hundredth of its density.
const FOG_RATE: f32 = 1000.0;
const FALLOFF_RATE: f32 = 20.0;
const DENSITY_RATE: f32 = 100.0;

/// What every shader in the engine weighs a colour's channels by to take its brightness.
const LUMA: Vec3 = Vec3::new(0.29891, 0.58661, 0.11448);

/// The lanes of the ambient entry no file states: what the sky harmonics come back up by, and the
/// scale and bias a sampled reflection takes against the term the frame picks, which every captured
/// frame holds at these.
const SKY_SCALE: f32 = 1.0;
const REFLECTION: Vec3 = Vec3::X;

/// Which of the reflection array's cubes a place stands under, which is the env map the
/// `EnvLocation` it is bound to names. Only the array variant reads it, and a zone draws the single.
const CAPTURE: f32 = 0.0;

/// One channel's harmonics, as a file states them.
type Channels = [[f32; 9]; 3];

/// A file the panel reads light out of.
enum Held<T> {
    Idle,
    Wanted(String),
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(T),
    Failed,
}

impl<T: File> Held<T> {
    fn poll(&mut self, backend: &Backend) {
        *self = match std::mem::replace(self, Self::Idle) {
            Self::Wanted(path) => {
                let files = backend.files().clone();
                Self::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&path).await
                }))
            }
            Self::Fetching(promise) => match promise.try_get().map(|held| match held {
                Ok(bytes) => Ok(bytes.clone()),
                Err(why) => Err(why.to_string()),
            }) {
                Some(Ok(bytes)) => match T::read(Cursor::new(bytes)) {
                    Ok(held) => Self::Ready(held),
                    Err(why) => {
                        log::warn!("assets/layer: {why}");
                        Self::Failed
                    }
                },
                Some(Err(_)) => Self::Failed,
                None => Self::Fetching(promise),
            },
            held => held,
        };
    }

    fn ready(&self) -> Option<&T> {
        match self {
            Self::Ready(held) => Some(held),
            _ => None,
        }
    }

    fn pending(&self) -> bool {
        matches!(self, Self::Wanted(_) | Self::Fetching(_))
    }

    fn state(&self) -> &'static str {
        match self {
            Self::Ready(_) => "read",
            Self::Failed => "could not be read",
            Self::Idle => "none",
            _ => "loading",
        }
    }
}

/// What the environment's cloud set states: the texture each mesh reads, and the light both are
/// drawn under.
pub struct Clouds {
    pub band: Option<u16>,
    pub sheet: Option<u16>,
    pub scene: program::Cloud,
}

/// A box, ellipsoid or cylinder a zone lights out of its own environment rather than out of the one
/// over the whole zone: the roofed parts of a town, and the like.
pub struct Space {
    /// Where it stands and how far it reaches, which its own instance states as a transform.
    pub placement: Mat4,
    /// The file's own shape code, which is what the composite compares against.
    pub shape: f32,
    /// How far in from a face the environment takes over, in the world's units.
    pub range: f32,
    /// The `EnvLocation` it takes its light from.
    pub bound: u32,
}

impl Space {
    /// Whether a place stands inside it. The shape codes are the file's own, the same three the
    /// composite tests each pixel against.
    fn holds(&self, at: Vec3) -> bool {
        let local = self.placement.inverse().transform_point3(at);
        match self.shape.to_bits() {
            1 => local.length() <= 1.0,
            3 => local.y.abs() <= 1.0 && Vec2::new(local.x, local.z).length() <= 1.0,
            _ => local.abs().cmple(Vec3::ONE).all(),
        }
    }
}

/// One of the environments a scene applies over part of itself.
struct Environment {
    envb: String,
    /// The `EnvLocation` instance the environment is centered on, and the `.amb` it names once a
    /// walk has reached the layer group holding it.
    instance: u32,
    amb: Option<String>,
}

/// What one weather of an `.envb` states at one time of day.
#[derive(Default)]
struct Lighting {
    stated: bool,
    sunlight: Vec3,
    moonlight: Vec3,
    /// What the background mixes its ambient toward, and how far.
    extra: Vec4,
    scale: f32,
    saturation: f32,
    attenuation: f32,
    /// How far down the ambient is allowed to fade with depth.
    floor: f32,
}

pub struct Ambient {
    environments: Vec<Environment>,
    at: usize,
    /// The environment the eye last stood in, so a hand-picked one is not taken away every frame.
    stood: Option<usize>,
    /// Which environment the loaded files belong to, so moving the picker fetches again.
    loaded: Option<usize>,
    weather_file: Held<envb::EnvironmentFile>,
    /// One per environment the scene names, since a zone lights its roofed parts out of their own
    /// files rather than out of the one the whole zone stands under.
    locations: Vec<Held<AmbientFile>>,
    sky_file: Held<AmbientFile>,

    /// Seconds since midnight.
    pub time: f32,
    /// Which of the `.amb`'s tracks the ambient is taken from. The game reads the first: a real
    /// frame's harmonics match track 0 two orders of magnitude closer than the next.
    pub track: usize,
    pub weather: usize,
    /// How far the sun's circle leans, which the scene's own level file states.
    pub tilt: f32,
    /// How far down the view the sun's own depth maps reach, which that file states beside it.
    pub reach: f32,
    /// How far up the frame the moon reaches, which no file states either.
    pub moon: f32,
    /// The moon's own day, `1..=32`, which no file states either: a date to stand the panel at
    /// rather than anything the hour derives.
    pub day: f32,
    /// The places inside the zone that light themselves, as the walk found them.
    pub spaces: Vec<Space>,
}

impl Ambient {
    pub fn new(scene: Option<&layer::Scene>) -> Self {
        // The zone's own lean, where it names a scene. A place with none stands under the one most
        // of them state.
        let tilt = scene.map_or(program::TILT, |held| held.sun_tilt_degrees() as f32);
        // Slot nine of the same block, which the cascades of five captures of three zones end at to
        // the digit. Not named by the parse yet, so read raw.
        let reach = scene
            .and_then(|held| held.general().get(SHADOW_REACH).copied())
            .map(f32::from_bits)
            .filter(|held| held.is_finite() && *held > 0.0)
            .unwrap_or(program::SHADOW_REACH);
        let environments = scene
            .map(|held| {
                held.environments()
                    .iter()
                    .filter(|env| !env.asset_path().is_empty())
                    .map(|env| Environment {
                        envb: env.asset_path().clone(),
                        instance: env.env_location_instance_id() as u32,
                        amb: None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            environments,
            at: 0,
            stood: None,
            loaded: None,
            weather_file: Held::Idle,
            locations: Vec::new(),
            sky_file: Held::Idle,
            time: DAY / 2.0,
            track: 0,
            weather: 0,
            tilt,
            reach,
            moon: MOON,
            day: FULL,
            spaces: Vec::new(),
        }
    }

    /// The `.amb` an `EnvLocation` instance names, taken as a walk reaches it.
    pub fn locate(&mut self, instance: u32, path: &str) {
        if path.is_empty() {
            return;
        }
        if let Some(env) = self
            .environments
            .iter_mut()
            .find(|env| env.instance == instance)
        {
            env.amb = Some(path.to_owned());
            return;
        }
        // The scene's own list holds what the whole zone stands under. A roofed place is bound to an
        // instance that list never names, and only a walk reaching its layer group finds it, so it
        // joins here: Ishgard states one environment and places thirty-odd of these.
        self.environments.push(Environment {
            envb: String::new(),
            instance,
            amb: Some(path.to_owned()),
        });
    }

    pub fn pending(&self) -> bool {
        self.weather_file.pending()
            || self.locations.iter().any(Held::pending)
            || self.sky_file.pending()
    }

    pub fn poll(&mut self, backend: &Backend) {
        // Only where the environment names a weather of its own: one the walk found names an `.amb`
        // and nothing else, and standing on it must not take the zone's weather away.
        if let Some(env) = self.environments.get(self.at)
            && self.loaded != Some(self.at)
            && !env.envb.is_empty()
        {
            self.weather_file = Held::Wanted(env.envb.clone());
            self.locations.clear();
            self.loaded = Some(self.at);
        }
        // Every environment, not only the one the panel stands in: each roofed place lights itself
        // out of its own file, and the `.amb` is named by an instance rather than by the scene, so
        // one only arrives once a walk has reached the layer group holding it.
        self.locations.resize_with(self.environments.len(), || Held::Idle);
        for (at, env) in self.environments.iter().enumerate() {
            if let (Some(Held::Idle), Some(path)) = (self.locations.get(at), &env.amb) {
                self.locations[at] = Held::Wanted(path.clone());
            }
        }
        if self.sky().is_some() && matches!(self.sky_file, Held::Idle) {
            self.sky_file = Held::Wanted(SKY_LIGHT.to_owned());
        }
        self.weather_file.poll(backend);
        for held in &mut self.locations {
            held.poll(backend);
        }
        self.sky_file.poll(backend);
        // Which tracks a file holds is the file's own business, and a track it states no keyframes
        // for samples nothing at all: the ambient would come out at nought everywhere.
        let tracks = self.tracks();
        if !tracks.is_empty() && !tracks.iter().any(|(held, _)| *held == self.track) {
            self.track = tracks[0].0;
        }
    }

    /// The tracks the location's `.amb` holds keyframes for, and how many each holds.
    fn tracks(&self) -> Vec<(usize, usize)> {
        let Some(AmbientFile::EnvLocation(held)) =
            self.locations.get(self.at).and_then(Held::ready)
        else {
            return Vec::new();
        };
        (0..TRACK_COUNT)
            .filter_map(|track| {
                let count = held.track(track)?.len();
                (count > 0).then_some((track, count))
            })
            .collect()
    }

    /// Which sky the frame stands under, which the weather states as its own parameter rather than
    /// in any of its sets. Measured: `s1f2` weather 1 states 273 and weather 7 states 3, and a
    /// capture of each binds `sky_273.tex` and `sky_003.tex` byte for byte. Nought is a weather that
    /// stands under no sky at all - no such file exists.
    pub fn sky(&self) -> Option<u16> {
        let held = self.weather_file.ready()?;
        let weather = held.environments().weathers().get(self.weather)?;
        u16::try_from(weather.parameter()).ok().filter(|id| *id > 0)
    }

    /// The id of the weather the panel stands in, as a preset states one.
    pub fn weather_id(&self) -> Option<u32> {
        self.weathers().get(self.weather).copied()
    }

    fn weathers(&self) -> Vec<u32> {
        self.weather_file
            .ready()
            .map(|held| {
                held.environments()
                    .weathers()
                    .iter()
                    .map(ironworks::file::envs::Weather::id)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The keyframes either side of the time the panel stands at, in the set of one kind of the
    /// weather it stands in, and how far between them it falls.
    fn keyframes(&self, kind: u32) -> Option<Between<'_>> {
        let set = self
            .weather_file
            .ready()?
            .environments()
            .weathers()
            .get(self.weather)?
            .sets()
            .iter()
            .find(|set| set.kind() == kind)?;
        let held = set.keyframes();
        let times: Vec<f32> = held.iter().map(Keyframe::time).collect();
        let (before, after, share) = between(&times, self.time)?;
        Some((&held[before], &held[after], share))
    }

    /// Takes up the environment of the volume the eye stands in, which is what the game picks by
    /// rather than by a list. Only where that volume changes, so a hand-picked one stays until the
    /// camera leaves the place it was picked for.
    pub fn stand_in(&mut self, eye: Vec3) {
        let bound = self
            .spaces
            .iter()
            .rev()
            .find(|space| space.holds(eye))
            .map(|space| space.bound);
        let at = bound
            .and_then(|bound| {
                self.environments
                    .iter()
                    .position(|env| env.instance == bound)
            })
            .unwrap_or_default();
        if self.stood != Some(at) {
            self.stood = Some(at);
            self.at = at;
        }
    }

    /// What the `.envb` states for the weather and time the panel stands at.
    fn lighting(&self) -> Lighting {
        let mut out = Lighting {
            scale: 1.0,
            ..Default::default()
        };
        let Some(held) = self.keyframes(GLOBAL_LIGHTING) else {
            return out;
        };
        let color = |name: &str| colour(held, name).map_or(Vec3::ZERO, |(held, _)| held);
        out.stated = true;
        out.sunlight = color("sunlight_color");
        out.moonlight = color("moonlight_color");
        // A color and the weight it carries, which is the shape the background's own ambient
        // parameter has and the only pair in the file with it.
        out.extra = colour(held, "extra_ambient_color")
            .map(|(held, _)| held)
            .unwrap_or(Vec3::ZERO)
            .extend(scalar(held, "extra_ambient_color_weight", 0.0));
        out.scale = scalar(held, "ambient_light_scale", 1.0);
        out.saturation = scalar(held, "ambient_light_saturation", 1.0);
        out.attenuation = scalar(held, "ambient_attenuation", 0.0);
        out.floor = scalar(held, "parameter_1", 0.0);
        out
    }

    /// What the exposure chain is run with, and nothing where the weather states no tone mapping at
    /// all: a frame with no numbers of its own is left as the composite resolved it rather than
    /// exposed against numbers invented here.
    pub fn exposure(&self, step: f32) -> Option<program::Exposure> {
        let held = self.keyframes(TONE_MAPPING)?;
        Some(program::Exposure {
            min: scalar(held, "adapted_luminance_parameter_x", 1.0),
            max: scalar(held, "adapted_luminance_parameter_y", 1.0),
            rate: scalar(held, "adaptation_rate", 0.0),
            key: scalar(held, "adapted_luminance_parameter_w", 1.0),
            strength: scalar(held, "tone_map_parameter_x", 0.0),
            shoulder: scalar(held, "tone_map_parameter_y", 0.0),
            step,
            ..Default::default()
        })
    }

    /// What share of a surface the composite counts as glare, which is what the bright pass weighs a
    /// pixel by. The engine takes both out of the wetness set, whose first two lanes reach
    /// `g_CommonParameter.m_Misc` unchanged: three frames measured reproduce from them exactly.
    pub fn bloom(&self) -> Option<program::Bloom> {
        let held = self.keyframes(WETNESS)?;
        Some(program::Bloom {
            specular: scalar(held, "world_wetness_parameter_0", 0.0),
            emissive: scalar(held, "world_wetness_parameter_1", 0.0),
        })
    }

    /// Stands the panel in the weather the id names, where the environment states one. A capture
    /// names its weather by id, and the picker holds them in whatever order the file does.
    pub fn stand_in_weather(&mut self, id: u32) -> bool {
        match self.weathers().iter().position(|held| *held == id) {
            Some(at) => {
                self.weather = at;
                true
            }
            None => false,
        }
    }

    /// What the weather says its moon looks like, and how much of the hour lets it through. The
    /// starfield set states an alpha of nought right through the day, which is what keeps the disc
    /// off the sky between five and eighteen without anything here deciding an hour.
    pub fn moonlight(&self) -> Vec4 {
        let Some(held) = self.keyframes(STARFIELD) else {
            return Vec4::ZERO;
        };
        let Some((color, alpha)) = colour(held, "moon_color") else {
            return Vec4::ZERO;
        };
        // A place stating no color for its moon draws none at all. The disc writes over the sky
        // rather than blending into it, so a black one would cut a hole in the stars.
        match color.max_element() > 0.0 {
            true => color.extend(alpha),
            false => Vec4::ZERO,
        }
    }

    /// How far the alpha falls off across the disc, which the starfield set states beside the
    /// moon's own color.
    pub fn moon_fade(&self) -> f32 {
        self.keyframes(STARFIELD).map_or(MOON_FADE, |held| scalar(held, "unknown", MOON_FADE))
    }

    /// What the point-star, Milky Way and instanced draws are run with, and nothing where the
    /// weather states no starfield set at all. `unknown` is tier 0's own flat output alpha, so a
    /// weather that states nought there is one the game draws nothing of.
    pub fn starfield(&self) -> Option<program::Star> {
        let held = self.keyframes(STARFIELD)?;
        Some(program::Star {
            horizon: scalar(held, "a_intensity", 0.0),
            point: scalar(held, "b_intensity", 0.0),
            band: scalar(held, "c_intensity", 0.0),
            alpha: scalar(held, "unknown", 0.0),
        })
    }

    /// What a leaf is swayed by, and nothing where the weather states no wind set. The set names two
    /// layers, each a heading and a strength; `bg.shpk`'s own buffer holds one heading, so the two
    /// sum there, but `grass.shpk`'s `g_WindInfo` keeps a texture-sampled strength per layer, so
    /// [`layers`](program::Wind::layers) carries them apart as well.
    ///
    /// The azimuth turns the other way from the plain reading: a capture's own `g_WindInfo` holds
    /// `(-sin, 0, cos)` of both azimuths this zone states, neither of which the plain reading fits.
    ///
    /// Each layer's `wavelength` feeds `grass.shpk`'s own world-to-texel scale for sampling
    /// `bgcommon/nature/wind/texture/wind_00{1,2}.tex`, as `1.0 / wavelength` with no term beside
    /// it. `min_strength` is read now that it has a real consumer (the same texture sample, squared,
    /// lerped between it and `max_strength`) rather than the naive time-based gust an earlier
    /// reading tried and reverted for freezing solid every cycle.
    pub fn wind(&self) -> Option<program::Wind> {
        let held = self.keyframes(WIND)?;
        let layer = |which: usize| {
            let of = |field: &str| scalar(held, &format!("layer_{which}_{field}"), 0.0);
            let heading = of("azimuth_degrees").to_radians();
            program::WindLayer {
                heading: Vec3::new(-heading.sin(), 0.0, heading.cos()),
                max_strength: of("max_strength"),
                min_strength: of("min_strength"),
                wavelength: of("wavelength"),
            }
        };
        let layers = [layer(0), layer(1)];
        let vector = |held: program::WindLayer| {
            glam::Vec2::new(held.heading.x, held.heading.z) * held.max_strength
        };
        let held = vector(layers[0]) + vector(layers[1]);
        Some(program::Wind {
            heading: Vec3::new(held.x, 0.0, held.y).normalize_or_zero(),
            reach: held.length(),
            layers,
        })
    }

    /// What the overlays a zone places carry, and nothing where the weather states no set for them.
    /// One set holds both: a shaft of light takes the first color, and a slab of fog takes the pair
    /// as its own surface and far below it, thickening at the rate stated beside them.
    pub fn shafts(&self) -> Option<program::Shaft> {
        let held = self.keyframes(LIGHT_SHAFT)?;
        Some(program::Shaft {
            color: colour(held, "color_0")?.0,
            radiance: colour(held, "radiance_color")?.0,
            scale: scalar(held, "scale", 0.0),
        })
    }

    /// What the cloud draws are run with, and nothing where the weather states no cloud set. A
    /// weather names a texture for each mesh, and a sheet of nought is one that draws none: no such
    /// file exists.
    pub fn clouds(&self) -> Option<Clouds> {
        let held = self.keyframes(CLOUDS)?;
        let of = |name: &str| colour(held, name).map_or(Vec3::ONE, |(held, _)| held);
        Some(Clouds {
            band: unsigned(held, "alt_cloud").filter(|id| *id > 0),
            sheet: unsigned(held, "main_cloud").filter(|id| *id > 0),
            scene: program::Cloud {
                diffuse: of("diffuse_color"),
                ambient: of("ambient_color"),
                // The band's own reach. Every frame measured states nine tenths for it, and every
                // one of them is a weather whose `main_intensity` is nine tenths too, so a fixed
                // nine tenths is not ruled out; the other field it could be reads one in all three.
                reach: scalar(held, "main_intensity", 0.9),
            },
        })
    }

    /// What the fog pass is run with, and nothing where the weather states no fog set or states one
    /// that never thickens: the table would hold one value the whole way out and the pass would leave
    /// every pixel where it found it.
    pub fn fog(&self) -> Option<program::Fog> {
        let held = self.keyframes(VERTICAL_FOG)?;
        let (color, cap) = colour(held, "fog_color")?;
        // The height layers are stated per twenty units of height and per hundred of density, and
        // the second layer's height as a step off the first's.
        let base = scalar(held, "exp_fog_height", 0.0);
        let layer = |falloff: &str, density: &str, height: f32| {
            Vec3::new(
                scalar(held, falloff, 0.0) / FALLOFF_RATE,
                scalar(held, density, 0.0) / DENSITY_RATE,
                height,
            )
        };
        let out = program::Fog {
            color,
            cap,
            rate: scalar(held, "fog_intensity_0", 0.0) / FOG_RATE,
            blend: scalar(held, "fog_intensity_1", 0.0),
            start: scalar(held, "fog_start_distance", 0.0),
            fade: scalar(held, "fog_fade_distance", 0.0),
            haze: switch(held, "use_height_fog_update"),
            near: scalar(held, "start_distance", 0.0),
            layers: [
                layer("fog_height_falloff", "fog_density_percent", base),
                layer(
                    "fog_height_falloff_2",
                    "fog_density_2_percent",
                    base + scalar(held, "exp_fog_height_2_delta", 0.0),
                ),
            ],
            clear: scalar(held, "fog_min_opacity", 0.0),
            glow: colour(held, "directional_inscattering_color").map_or(Vec3::ZERO, |(held, _)| held),
            glow_strength: scalar(held, "directional_inscattering_color_intensity", 0.0),
            glow_sharpness: scalar(held, "directional_inscattering_exponent", 0.0),
            glow_start: scalar(held, "directional_inscattering_start_distance", 0.0),
        };
        let thickens = out.rate > 0.0 && out.cap > 0.0;
        let hazes = out.haze > 0.0 && out.layers.iter().any(|held| held.y > 0.0);
        (thickens || hazes).then_some(out)
    }

    /// The key light: which way it comes from, which nothing states, and the color the zone does.
    /// The sun and the moon go in together, since the graph runs one directional pass and a zone
    /// cross-fades the two over the day.
    pub fn light(&self) -> (Vec3, Vec3) {
        let held = self.lighting();
        let color = match held.stated {
            true => held.sunlight + held.moonlight,
            false => Vec3::ONE,
        };
        // Whichever of the two stands above the horizon. After dark a capture puts the key at the
        // moon's azimuth rather than the sun's, so it does turn around. Both are read higher up
        // than the body they follow, by an amount no shipped file states and no rule fits across
        // zones, which is why only the side is taken here and not the lift.
        let sun = program::sun(self.time, self.tilt);
        let toward = match sun.y >= 0.0 {
            true => sun,
            false => program::moon(self.time, self.tilt),
        };
        (toward, color)
    }

    /// The ambient buffer as the files decide it.
    pub fn scene(&self) -> program::Ambient {
        let held = self.lighting();
        program::Ambient {
            sky: rows(self.sky_light()),
            sky_scale: SKY_SCALE,
            light: greyer(rows(self.harmonics()), held.saturation),
            scale: held.scale,
            fade: Vec3::new(0.0, 1.0, held.floor),
            reflection: REFLECTION,
            capture: CAPTURE,
            volumes: self.volumes(),
        }
    }

    /// One array entry per place that lights itself, each with the harmonics of the environment it
    /// is bound to. A space whose own light has not arrived is left out rather than drawn dark.
    fn volumes(&self) -> std::sync::Arc<[program::Volume]> {
        let held = self.lighting();
        let (scale, saturation) = (held.scale, held.saturation);
        self.spaces
            .iter()
            .filter_map(|space| {
                let at = self
                    .environments
                    .iter()
                    .position(|env| env.instance == space.bound)?;
                let light = greyer(rows(Some(self.harmonics_of(at)?)), saturation);
                // The composite takes a place in front of the camera into the volume's own space,
                // where it stands as the unit shape, so the placement's own scale is its extent.
                let (size, _, _) = space.placement.to_scale_rotation_translation();
                // How sharply it takes over, in units of that extent: a range the file states in the
                // world is a fraction of the half extent it is measured across.
                let fade = Vec3::new(
                    size.x / space.range.max(0.001),
                    size.y / space.range.max(0.001),
                    size.z / space.range.max(0.001),
                );
                Some(program::Volume {
                    into: space.placement.inverse(),
                    fade: fade.clamp(Vec3::ONE, Vec3::splat(64.0)),
                    shape: space.shape,
                    light,
                    scale,
                })
            })
            .collect()
    }

    /// The location's ambient light at the time the panel stands at.
    fn harmonics(&self) -> Option<Channels> {
        self.harmonics_of(self.at)
    }

    /// The harmonics one of the scene's environments states at that time.
    fn harmonics_of(&self, at: usize) -> Option<Channels> {
        let AmbientFile::EnvLocation(held) = self.locations.get(at).and_then(Held::ready)? else {
            return None;
        };
        let keyframes = held.track(self.track)?;
        let times: Vec<f32> = keyframes.iter().map(amb::Keyframe::time).collect();
        let (before, after, share) = between(&times, self.time)?;
        Some(blend(
            channels(keyframes[before].light()),
            channels(keyframes[after].light()),
            share,
        ))
    }

    /// The sky's own light at that time. The file holds no time per sample, so its samples are read
    /// as an even run over the day, which for the twenty-four of them the skies mostly carry is an
    /// hour apiece.
    fn sky_light(&self) -> Option<Channels> {
        let AmbientFile::SkyLight(held) = self.sky_file.ready()? else {
            return None;
        };
        let samples = held.samples(self.sky()?)?;
        if samples.is_empty() {
            return None;
        }
        let step = DAY / samples.len() as f32;
        let times: Vec<f32> = (0..samples.len()).map(|at| at as f32 * step).collect();
        let (before, after, share) = between(&times, self.time)?;
        Some(blend(
            channels(samples[before]),
            channels(samples[after]),
            share,
        ))
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        follow: &mut Option<String>,
        deps: &mut Deps,
        backend: &Backend,
    ) -> bool {
        let mut changed = false;
        section(ui, "Environment");
        if self.environments.len() > 1 {
            ui.label(RichText::new("Environment").weak());
            egui::ComboBox::from_id_salt("scene_environment")
                .truncate()
                .selected_text(format!("{} of {}", self.at + 1, self.environments.len()))
                .show_ui(ui, |ui| {
                    for at in 0..self.environments.len() {
                        changed |= ui
                            .selectable_value(&mut self.at, at, format!("{}", at + 1))
                            .changed();
                    }
                });
        }

        let hours = self.time / 3600.0;
        ui.label(
            RichText::new(format!(
                "Time of day  {:02}:{:02}",
                hours as u32,
                (hours.fract() * 60.0) as u32
            ))
            .weak(),
        );
        changed |= ui
            .add(egui::Slider::new(&mut self.time, 0.0..=DAY).show_value(false))
            .changed();

        ui.label(
            RichText::new(format!(
                "Day {}  {}",
                self.day.round() as u32,
                program::moon_phase_name(self.day)
            ))
            .weak(),
        );
        changed |= ui
            .add(egui::Slider::new(&mut self.day, 1.0..=32.0).step_by(1.0).show_value(false))
            .changed();

        let weathers = self.weathers();
        if !weathers.is_empty() {
            let mut named = |ui: &egui::Ui, id: u32| match deps.text(ui.ctx(), backend, WEATHER, id)
            {
                Some(name) => format!("{id}  {name}"),
                None => id.to_string(),
            };
            ui.label(RichText::new("Weather").weak());
            let selected = weathers
                .get(self.weather)
                .map_or_else(String::new, |id| named(ui, *id));
            egui::ComboBox::from_id_salt("scene_weather")
                .truncate()
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (at, id) in weathers.iter().enumerate() {
                        let label = named(ui, *id);
                        changed |= ui.selectable_value(&mut self.weather, at, label).changed();
                    }
                });
        }

        if self.moonlight().w > 0.0 {
            ui.label(
                RichText::new(format!("Moon  {:.3} deg across", self.moon.atan().to_degrees() * 2.0))
                    .weak(),
            );
            changed |= ui
                .add(egui::Slider::new(&mut self.moon, 0.0002..=0.05).logarithmic(true).show_value(false))
                .changed();
        }

        ui.add_space(8.0);
        let env = self.environments.get(self.at);
        let files = [
            (
                "Ambient",
                self.locations.get(self.at).map_or("none", Held::state),
                env.and_then(|held| held.amb.clone()),
            ),
            (
                "Weather",
                self.weather_file.state(),
                env.map(|held| held.envb.clone()),
            ),
        ];
        let held = self.lighting();
        let mut rows = Vec::new();
        if held.stated {
            let spell = |held: Vec3| format!("{:.3}, {:.3}, {:.3}", held.x, held.y, held.z);
            rows.extend([
                ("Sunlight", spell(held.sunlight)),
                ("Moonlight", spell(held.moonlight)),
                ("Key light", spell(held.sunlight + held.moonlight)),
                ("Key direction", spell(self.light().0)),
                ("Sun tilt", format!("{:.0} deg", self.tilt)),
                ("Shadow reach", format!("{:.0}", self.reach)),
                (
                    "Sky",
                    match self.sky() {
                        Some(id) => format!("{id:03}"),
                        None => "none".to_owned(),
                    },
                ),
                (
                    "Extra ambient",
                    format!("{} at {:.3}", spell(held.extra.truncate()), held.extra.w),
                ),
                ("Ambient track", self.track.to_string()),
                ("Ambient light scale", format!("{:.3}", held.scale)),
                ("Saturation", format!("{:.3}", held.saturation)),
                ("Attenuation", format!("{:.3}", held.attenuation)),
                ("Depth fade", spell(Vec3::new(0.0, 1.0, held.floor))),
            ]);
        }
        rows.extend([
            (
                "Ambient volumes",
                format!("{} of {} placed", self.volumes().len(), self.spaces.len()),
            ),
            ("Sky harmonic scale", format!("{SKY_SCALE:.3}")),
            (
                "Reflection",
                format!(
                    "{:.3}, {:.3}, {:.3}",
                    REFLECTION.x, REFLECTION.y, REFLECTION.z
                ),
            ),
            ("Reflection capture", format!("{CAPTURE:.0}")),
        ]);
        // What the frame is exposed and read back through, which moves with the weather and the hour
        // the same way the light above it does. A zone that states none of this is left alone.
        if let Some(held) = self.exposure(0.0) {
            rows.extend([
                ("Exposure range", format!("{:.3} to {:.3}", held.min, held.max)),
                ("Exposure key", format!("{:.3}", held.key)),
                ("Adaptation rate", format!("{:.3}/s", held.rate)),
                ("Tone curve", format!("{:.3} at {:.3}", held.strength, held.shoulder)),
            ]);
        }
        // The clouds, whose two textures the weather names by id and whose colors it states beside
        // them. A mesh the weather names nothing for is not drawn at all.
        if let Some(held) = self.clouds() {
            let named = |id: Option<u16>| match id {
                Some(id) => format!("{id:03}"),
                None => "none".to_owned(),
            };
            let spell = |held: Vec3| format!("{:.3}, {:.3}, {:.3}", held.x, held.y, held.z);
            rows.extend([
                (
                    "Clouds",
                    format!("band {}  sheet {}", named(held.band), named(held.sheet)),
                ),
                ("Cloud lit", spell(held.scene.diffuse)),
                ("Cloud shaded", spell(held.scene.ambient)),
                ("Cloud reach", format!("{:.3}", held.scene.reach)),
            ]);
        }
        // The same for the fog, whose every number is the weather's own.
        if let Some(held) = self.fog() {
            rows.extend([
                (
                    "Fog color",
                    format!(
                        "{:.3}, {:.3}, {:.3} at {:.3}",
                        held.color.x, held.color.y, held.color.z, held.cap
                    ),
                ),
                ("Fog start", format!("{:.0} to {:.0}", held.start, held.far())),
                ("Fog fade", format!("{:.0}", held.fade)),
                (
                    "Haze",
                    match held.haze > 0.0 {
                        true => format!("from {:.0}, leaving {:.3}", held.near, held.clear),
                        false => "off".to_owned(),
                    },
                ),
            ]);
            rows.extend(held.layers.iter().enumerate().map(|(at, layer)| {
                (
                    match at {
                        0 => "Haze layer",
                        _ => "Haze layer 2",
                    },
                    format!(
                        "{:.5} thick at {:.0}, thinning {:.5}",
                        layer.y, layer.z, layer.x
                    ),
                )
            }));
            rows.push((
                "Haze glow",
                format!(
                    "{:.3}, {:.3}, {:.3} at {:.2} to the {:.0} from {:.0}",
                    held.glow.x,
                    held.glow.y,
                    held.glow.z,
                    held.glow_strength,
                    held.glow_sharpness,
                    held.glow_start
                ),
            ));
        }
        ui.scope(|ui| {
            ui.set_max_width(ui.available_width().min(super::DETAILS_ROW_WIDTH));
            egui::Grid::new("scene_environment_files")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    for (label, state, path) in files {
                        ui.label(RichText::new(label).weak());
                        match &path {
                            Some(path) => {
                                if link(ui, crate::utils::file_name(path), path) {
                                    *follow = Some(path.clone());
                                }
                            }
                            None => {
                                ui.add(egui::Label::new(RichText::new(state).monospace()).wrap());
                            }
                        }
                        ui.allocate_space(egui::vec2(ui.available_width(), 0.0));
                        ui.end_row();
                    }
                    for (label, value) in &rows {
                        ui.label(RichText::new(*label).weak());
                        ui.add(egui::Label::new(RichText::new(value).monospace()).wrap());
                        ui.allocate_space(egui::vec2(ui.available_width(), 0.0));
                        ui.end_row();
                    }
                });
        });
        changed
    }
}

/// The keyframes either side of a time, and how far between them it falls.
type Between<'a> = (&'a Keyframe, &'a Keyframe, f32);

/// One field of both keyframes, where both carry it.
fn field<'a>(held: Between<'a>, name: &str) -> Option<(&'a Value, &'a Value)> {
    let of = |keyframe: &'a Keyframe| {
        keyframe
            .fields()
            .iter()
            .find(|(field, _)| *field == name)
            .map(|(_, value)| value)
    };
    of(held.0).zip(of(held.1))
}

/// One of its colours, interpolated: the channels as fractions of full, taken up by the intensity
/// the file states beside them, and the alpha byte the same way but never scaled.
fn colour(held: Between<'_>, name: &str) -> Option<(Vec3, f32)> {
    let (Value::Colour(first), Value::Colour(second)) = field(held, name)? else {
        return None;
    };
    let of = |held: &ironworks::file::envs::Colour| {
        (
            Vec3::new(
                f32::from(held.red()),
                f32::from(held.green()),
                f32::from(held.blue()),
            ) / 255.0
                * held.intensity(),
            f32::from(held.alpha()) / 255.0,
        )
    };
    let (first, second) = (of(first), of(second));
    Some((
        first.0.lerp(second.0, held.2),
        first.1 + (second.1 - first.1) * held.2,
    ))
}

/// One of its whole numbers, taken off the keyframe the time has passed rather than interpolated:
/// what these name is a file, and there is no file between two of them.
fn unsigned(held: Between<'_>, name: &str) -> Option<u16> {
    match field(held, name)? {
        (Value::Unsigned(held), _) => u16::try_from(*held).ok(),
        _ => None,
    }
}

/// One of its floats, interpolated. A set that does not carry the field answers with the fallback,
/// since nought is a real setting for most of them.
fn scalar(held: Between<'_>, name: &str, fallback: f32) -> f32 {
    match field(held, name) {
        Some((Value::Float(first), Value::Float(second))) => {
            first + (second - first) * held.2
        }
        _ => fallback,
    }
}

/// A field the file states as a whole number rather than as a float, which the keyframe it falls in
/// holds outright: there is nothing between one and nought to cross.
fn switch(held: Between<'_>, name: &str) -> f32 {
    match field(held, name) {
        Some((Value::Unsigned(held), _)) => *held as f32,
        Some((Value::Flag(held), _)) => f32::from(*held),
        _ => 0.0,
    }
}

/// The harmonics taken toward grey by however far the weather's `ambient_light_saturation` says. A
/// row is one channel, so the three of them at a lane are the colour arriving from that direction.
fn greyer(held: [Vec4; 3], saturation: f32) -> [Vec4; 3] {
    let grey = held[0] * LUMA.x + held[1] * LUMA.y + held[2] * LUMA.z;
    held.map(|row| grey + (row - grey) * saturation)
}

/// The three rows one set of harmonics reaches the shader as, with the linear terms weighted.
fn rows(held: Option<Channels>) -> [Vec4; 3] {
    let Some(held) = held else {
        return [Vec4::ZERO; 3];
    };
    held.map(|channel| program::Ambient::row(&channel))
}

fn channels(held: amb::Harmonics) -> Channels {
    [held.red(), held.green(), held.blue()]
}

fn blend(first: Channels, second: Channels, share: f32) -> Channels {
    std::array::from_fn(|channel| {
        std::array::from_fn(|at| {
            first[channel][at] + (second[channel][at] - first[channel][at]) * share
        })
    })
}

/// Where `time` falls among keyframes ascending by time: the two either side of it and how far it
/// stands between them. The day wraps, so midnight sits between the evening and the morning rather
/// than at one end of the run.
fn between(times: &[f32], time: f32) -> Option<(usize, usize, f32)> {
    let last = times.len().checked_sub(1)?;
    if time < times[0] || time >= times[last] {
        let span = DAY - times[last] + times[0];
        let past = match time >= times[last] {
            true => time - times[last],
            false => DAY - times[last] + time,
        };
        return Some((last, 0, if span > 0.0 { past / span } else { 0.0 }));
    }
    let after = times.iter().position(|held| *held > time).unwrap_or(last);
    let span = times[after] - times[after - 1];
    Some((
        after - 1,
        after,
        match span > 0.0 {
            true => (time - times[after - 1]) / span,
            false => 0.0,
        },
    ))
}

/// One labelled slider over a field of the panel.
fn slider(ui: &mut egui::Ui, label: &str, held: &mut f32, range: RangeInclusive<f32>) -> bool {
    ui.label(RichText::new(label).weak());
    ui.add(egui::Slider::new(held, range)).changed()
}

#[cfg(test)]
mod test {
    use super::{DAY, between};

    #[test]
    fn a_time_falls_between_the_keyframes_either_side_of_it() {
        let times = [3600.0, 7200.0, 10800.0];
        assert_eq!(between(&times, 3600.0), Some((0, 1, 0.0)));
        assert_eq!(between(&times, 5400.0), Some((0, 1, 0.5)));
        assert_eq!(between(&times, 7200.0), Some((1, 2, 0.0)));
        assert_eq!(between(&[], 0.0), None);
    }

    #[test]
    fn the_day_wraps_past_the_last_keyframe() {
        let times = [3600.0, 10800.0];
        let (before, after, share) = between(&times, 0.0).unwrap();
        assert_eq!((before, after), (1, 0));
        assert!((share - (DAY - 10800.0) / (DAY - 10800.0 + 3600.0)).abs() < 1e-6);
        assert_eq!(
            between(&times, 10800.0).map(|held| (held.0, held.1)),
            Some((1, 0))
        );
    }
}
