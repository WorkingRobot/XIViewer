//! What a frame was drawn from, put where a harness measuring it can read it back.
//!
//! A screenshot alone cannot say which build drew it, where the scene's own rect sits inside the
//! window, or which camera it stood at, and every one of those has cost a wrong conclusion.

use serde::Serialize;

#[derive(Serialize)]
pub struct Frame<'a> {
    pub commit: &'a str,
    /// False where the tree had uncommitted work, so the commit alone does not name the build.
    pub clean: bool,
    pub built: &'a str,
    pub level: &'a str,
    pub preset: Option<&'a str>,
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    /// Vertical, in degrees, and what the projection matrix actually used: for a driven shot this
    /// is already refit to `viewport`'s aspect, not the shot's own 16:9 value.
    pub fov: f32,
    /// The scene's own rect inside the window, in physical pixels.
    pub viewport: [f32; 4],
    /// Seconds since midnight.
    pub time: f32,
    pub weather: u32,
    pub exposure: f32,
    pub measured: f32,
    /// How long the frame took, which is what the adaptation's per-second rate is scaled by.
    pub step: f32,
    pub placed: usize,
    pub drawn: usize,
    /// Placements collected, whether or not their file has arrived yet.
    pub effects: usize,
    /// Batches actually issued this frame, summed over every effect whose file and textures had
    /// arrived: proof a draw happened rather than only that a placement was collected.
    pub effects_drawn: usize,
    /// How many of those the sun's own pass draws, the rest stating that they cast nothing.
    pub casting: usize,
    pub models: String,
    pub materials: String,
    /// Lamps the frame lit with, of every light the zone places.
    pub lights: String,
    pub passes: String,
}

#[cfg(target_arch = "wasm32")]
pub fn publish(held: &Frame<'_>) {
    let Ok(text) = serde_json::to_string(held) else {
        return;
    };
    if let Some(window) = web_sys::window() {
        let _ = js_sys::Reflect::set(&window, &"__frame".into(), &text.into());
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn publish(_: &Frame<'_>) {}
