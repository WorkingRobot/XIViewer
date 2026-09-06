//! What the viewer says about the frame it drew, read back out of the page.

use glam::Vec3;
use serde::Deserialize;

use crate::View;

#[derive(Clone, Debug, Deserialize)]
pub struct State {
    /// The commit the wasm was built from, which is what stops a measurement against a stale build.
    pub commit: String,
    /// False where the tree had uncommitted work, so the commit alone does not name the build.
    pub clean: bool,
    pub built: String,
    pub level: String,
    pub preset: Option<String>,
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    /// Vertical, in degrees.
    pub fov: f32,
    /// The scene's own rect inside the window, in physical pixels.
    pub viewport: [f32; 4],
    /// Seconds since midnight.
    pub time: f32,
    pub weather: u32,
    pub exposure: f32,
    pub measured: f32,
    /// How long the frame took. The adaptation's rate is stated per second and the loop it closes
    /// only settles while the two multiply to under two thirds.
    pub step: f32,
    pub placed: usize,
    pub drawn: usize,
    pub models: String,
    pub materials: String,
    #[serde(default)]
    pub lights: String,
    pub passes: String,
}

impl State {
    pub fn view(&self) -> View {
        let [_, _, width, height] = self.viewport;
        View::of(
            Vec3::from_array(self.eye),
            Vec3::from_array(self.forward),
            self.fov,
            width.round() as u32,
            height.round() as u32,
        )
    }

    /// The clock the panel shows, which is what a preset states.
    pub fn clock(&self) -> String {
        let minutes = (self.time / 60.0).round() as u32;
        format!("{:02}:{:02}", minutes / 60 % 24, minutes % 60)
    }
}
